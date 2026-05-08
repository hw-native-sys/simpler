#include <arpa/inet.h>
#include <errno.h>
#include <infiniband/verbs.h>
#include <netinet/in.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

struct simpler_rxe_qp_info {
    uint32_t qpn;
    uint32_t psn;
    uint32_t rkey;
    uint64_t addr;
    uint32_t size;
    uint8_t gid[16];
};

struct simpler_rxe_server_desc {
    char ip[64];
    uint16_t port;
    uint32_t rkey;
    uint64_t addr;
    uint32_t size;
};

struct simpler_rxe_handle {
    struct ibv_context *ctx;
    struct ibv_pd *pd;
    struct ibv_cq *cq;
    struct ibv_qp *qp;
    struct ibv_mr *mr;
    pthread_t thread;
    void *addr;
    size_t size;
    char device[64];
    char ip[64];
    int gid_index;
    int listen_fd;
    int conn_fd;
    int ready_fd;
    uint16_t port;
    volatile int stop;
    volatile int rc;
    char err[256];
};

static void set_err(struct simpler_rxe_handle *h, const char *msg)
{
    if (h != NULL && msg != NULL) {
        snprintf(h->err, sizeof(h->err), "%s: errno=%d", msg, errno);
        h->rc = errno ? -errno : -1;
    }
}

static int send_all(int fd, const void *buf, size_t len)
{
    const char *p = (const char *)buf;
    while (len > 0) {
        ssize_t n = send(fd, p, len, 0);
        if (n <= 0) {
            return -1;
        }
        p += n;
        len -= (size_t)n;
    }
    return 0;
}

static int recv_all(int fd, void *buf, size_t len)
{
    char *p = (char *)buf;
    while (len > 0) {
        ssize_t n = recv(fd, p, len, 0);
        if (n <= 0) {
            return -1;
        }
        p += n;
        len -= (size_t)n;
    }
    return 0;
}

static struct ibv_context *open_device(const char *device)
{
    int num = 0;
    struct ibv_device **list = ibv_get_device_list(&num);
    if (list == NULL) {
        return NULL;
    }
    struct ibv_context *ctx = NULL;
    for (int i = 0; i < num; ++i) {
        const char *name = ibv_get_device_name(list[i]);
        if (name != NULL && strcmp(name, device) == 0) {
            ctx = ibv_open_device(list[i]);
            break;
        }
    }
    ibv_free_device_list(list);
    return ctx;
}

static int gid_from_context(struct ibv_context *ctx, int gid_index, uint8_t gid[16])
{
    union ibv_gid raw_gid;
    if (ibv_query_gid(ctx, 1, gid_index, &raw_gid) != 0) {
        return -1;
    }
    memcpy(gid, raw_gid.raw, 16);
    return 0;
}

static int modify_qp_init(struct ibv_qp *qp)
{
    struct ibv_qp_attr attr;
    memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_INIT;
    attr.port_num = 1;
    attr.pkey_index = 0;
    attr.qp_access_flags = IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_LOCAL_WRITE;
    return ibv_modify_qp(qp, &attr,
        IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS);
}

static int modify_qp_rtr(struct ibv_qp *qp, const struct simpler_rxe_qp_info *remote, int gid_index)
{
    struct ibv_qp_attr attr;
    memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_RTR;
    attr.path_mtu = IBV_MTU_1024;
    attr.dest_qp_num = remote->qpn;
    attr.rq_psn = remote->psn;
    attr.max_dest_rd_atomic = 1;
    attr.min_rnr_timer = 12;
    attr.ah_attr.is_global = 1;
    memcpy(attr.ah_attr.grh.dgid.raw, remote->gid, 16);
    attr.ah_attr.grh.sgid_index = gid_index;
    attr.ah_attr.grh.hop_limit = 1;
    attr.ah_attr.dlid = 0;
    attr.ah_attr.sl = 0;
    attr.ah_attr.src_path_bits = 0;
    attr.ah_attr.port_num = 1;
    return ibv_modify_qp(qp, &attr,
        IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
        IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER);
}

static int modify_qp_rts(struct ibv_qp *qp, uint32_t psn)
{
    struct ibv_qp_attr attr;
    memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_RTS;
    attr.timeout = 14;
    attr.retry_cnt = 7;
    attr.rnr_retry = 7;
    attr.sq_psn = psn;
    attr.max_rd_atomic = 1;
    return ibv_modify_qp(qp, &attr,
        IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY |
        IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC);
}

static int setup_verbs(struct simpler_rxe_handle *h)
{
    h->ctx = open_device(h->device);
    if (h->ctx == NULL) {
        set_err(h, "ibv_open_device failed");
        return -1;
    }
    h->pd = ibv_alloc_pd(h->ctx);
    if (h->pd == NULL) {
        set_err(h, "ibv_alloc_pd failed");
        return -1;
    }
    h->cq = ibv_create_cq(h->ctx, 16, NULL, NULL, 0);
    if (h->cq == NULL) {
        set_err(h, "ibv_create_cq failed");
        return -1;
    }
    struct ibv_qp_init_attr qp_init;
    memset(&qp_init, 0, sizeof(qp_init));
    qp_init.send_cq = h->cq;
    qp_init.recv_cq = h->cq;
    qp_init.qp_type = IBV_QPT_RC;
    qp_init.cap.max_send_wr = 16;
    qp_init.cap.max_recv_wr = 16;
    qp_init.cap.max_send_sge = 1;
    qp_init.cap.max_recv_sge = 1;
    h->qp = ibv_create_qp(h->pd, &qp_init);
    if (h->qp == NULL) {
        set_err(h, "ibv_create_qp failed");
        return -1;
    }
    h->mr = ibv_reg_mr(h->pd, h->addr, h->size,
        IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
    if (h->mr == NULL) {
        set_err(h, "ibv_reg_mr failed");
        return -1;
    }
    if (modify_qp_init(h->qp) != 0) {
        set_err(h, "modify_qp_init failed");
        return -1;
    }
    return 0;
}

static int listen_socket(struct simpler_rxe_handle *h)
{
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) {
        set_err(h, "socket failed");
        return -1;
    }
    int yes = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons(0);
    if (inet_pton(AF_INET, h->ip, &addr.sin_addr) != 1) {
        close(fd);
        set_err(h, "inet_pton failed");
        return -1;
    }
    if (bind(fd, (struct sockaddr *)&addr, sizeof(addr)) != 0) {
        close(fd);
        set_err(h, "bind failed");
        return -1;
    }
    socklen_t len = sizeof(addr);
    if (getsockname(fd, (struct sockaddr *)&addr, &len) != 0) {
        close(fd);
        set_err(h, "getsockname failed");
        return -1;
    }
    if (listen(fd, 1) != 0) {
        close(fd);
        set_err(h, "listen failed");
        return -1;
    }
    h->listen_fd = fd;
    h->port = ntohs(addr.sin_port);
    return 0;
}

static void *server_main(void *arg)
{
    struct simpler_rxe_handle *h = (struct simpler_rxe_handle *)arg;
    if (setup_verbs(h) != 0 || listen_socket(h) != 0) {
        if (h->ready_fd >= 0) {
            (void)write(h->ready_fd, "E", 1);
        }
        return NULL;
    }
    if (h->ready_fd >= 0) {
        (void)write(h->ready_fd, "R", 1);
    }
    int fd = accept(h->listen_fd, NULL, NULL);
    if (fd < 0) {
        if (!h->stop) {
            set_err(h, "accept failed");
        }
        return NULL;
    }
    h->conn_fd = fd;

    struct simpler_rxe_qp_info local;
    struct simpler_rxe_qp_info remote;
    memset(&local, 0, sizeof(local));
    memset(&remote, 0, sizeof(remote));
    local.qpn = h->qp->qp_num;
    local.psn = 0x111111;
    local.rkey = h->mr->rkey;
    local.addr = (uint64_t)(uintptr_t)h->addr;
    local.size = (uint32_t)h->size;
    if (gid_from_context(h->ctx, h->gid_index, local.gid) != 0) {
        set_err(h, "ibv_query_gid failed");
        close(fd);
        h->conn_fd = -1;
        return NULL;
    }
    if (send_all(fd, &local, sizeof(local)) != 0 || recv_all(fd, &remote, sizeof(remote)) != 0) {
        set_err(h, "server qp info exchange failed");
        close(fd);
        h->conn_fd = -1;
        return NULL;
    }
    if (modify_qp_rtr(h->qp, &remote, h->gid_index) != 0 || modify_qp_rts(h->qp, local.psn) != 0) {
        set_err(h, "server qp transition failed");
        close(fd);
        h->conn_fd = -1;
        return NULL;
    }
    char done = 0;
    if (recv_all(fd, &done, 1) != 0 || done != 'D') {
        set_err(h, "server completion wait failed");
    }
    close(fd);
    h->conn_fd = -1;
    return NULL;
}

int simpler_rxe_server_start(const char *device, int gid_index, const char *ip, void *addr, uint64_t size,
    struct simpler_rxe_server_desc *desc, void **out)
{
    if (device == NULL || ip == NULL || addr == NULL || size == 0 || desc == NULL || out == NULL) {
        return -EINVAL;
    }
    struct simpler_rxe_handle *h = (struct simpler_rxe_handle *)calloc(1, sizeof(*h));
    if (h == NULL) {
        return -ENOMEM;
    }
    snprintf(h->device, sizeof(h->device), "%s", device);
    snprintf(h->ip, sizeof(h->ip), "%s", ip);
    h->gid_index = gid_index;
    h->addr = addr;
    h->size = (size_t)size;
    h->listen_fd = -1;
    h->conn_fd = -1;
    h->ready_fd = -1;
    int pipefd[2];
    if (pipe(pipefd) != 0) {
        free(h);
        return -errno;
    }
    h->ready_fd = pipefd[1];
    if (pthread_create(&h->thread, NULL, server_main, h) != 0) {
        int rc = -errno;
        close(pipefd[0]);
        close(pipefd[1]);
        free(h);
        return rc;
    }
    char ready = 0;
    if (read(pipefd[0], &ready, 1) != 1 || ready != 'R') {
        int rc = h->rc ? h->rc : -1;
        pthread_join(h->thread, NULL);
        close(pipefd[0]);
        close(pipefd[1]);
        free(h);
        return rc;
    }
    close(pipefd[0]);
    close(pipefd[1]);
    h->ready_fd = -1;
    memset(desc, 0, sizeof(*desc));
    snprintf(desc->ip, sizeof(desc->ip), "%s", h->ip);
    desc->port = h->port;
    desc->rkey = h->mr->rkey;
    desc->addr = (uint64_t)(uintptr_t)h->addr;
    desc->size = (uint32_t)h->size;
    *out = h;
    return 0;
}

void simpler_rxe_server_stop(void *handle)
{
    struct simpler_rxe_handle *h = (struct simpler_rxe_handle *)handle;
    if (h == NULL) {
        return;
    }
    h->stop = 1;
    if (h->listen_fd >= 0) {
        shutdown(h->listen_fd, SHUT_RDWR);
        close(h->listen_fd);
        h->listen_fd = -1;
    }
    if (h->conn_fd >= 0) {
        shutdown(h->conn_fd, SHUT_RDWR);
    }
    pthread_join(h->thread, NULL);
    if (h->conn_fd >= 0) {
        close(h->conn_fd);
        h->conn_fd = -1;
    }
    if (h->mr != NULL) {
        ibv_dereg_mr(h->mr);
    }
    if (h->qp != NULL) {
        ibv_destroy_qp(h->qp);
    }
    if (h->cq != NULL) {
        ibv_destroy_cq(h->cq);
    }
    if (h->pd != NULL) {
        ibv_dealloc_pd(h->pd);
    }
    if (h->ctx != NULL) {
        ibv_close_device(h->ctx);
    }
    free(h);
}

int simpler_rxe_write(const char *device, int gid_index, const char *ip, uint16_t port,
    const void *local_addr, uint64_t size)
{
    if (device == NULL || ip == NULL || local_addr == NULL || size == 0) {
        return -EINVAL;
    }
    struct simpler_rxe_handle h;
    memset(&h, 0, sizeof(h));
    snprintf(h.device, sizeof(h.device), "%s", device);
    h.gid_index = gid_index;
    h.addr = (void *)local_addr;
    h.size = (size_t)size;
    h.listen_fd = -1;
    if (setup_verbs(&h) != 0) {
        goto fail;
    }
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) {
        goto fail;
    }
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    if (inet_pton(AF_INET, ip, &addr.sin_addr) != 1 || connect(fd, (struct sockaddr *)&addr, sizeof(addr)) != 0) {
        close(fd);
        goto fail;
    }

    struct simpler_rxe_qp_info local;
    struct simpler_rxe_qp_info remote;
    memset(&local, 0, sizeof(local));
    memset(&remote, 0, sizeof(remote));
    if (recv_all(fd, &remote, sizeof(remote)) != 0) {
        close(fd);
        goto fail;
    }
    local.qpn = h.qp->qp_num;
    local.psn = 0x222222;
    local.rkey = h.mr->rkey;
    local.addr = (uint64_t)(uintptr_t)local_addr;
    local.size = (uint32_t)size;
    if (gid_from_context(h.ctx, gid_index, local.gid) != 0) {
        close(fd);
        goto fail;
    }
    if (send_all(fd, &local, sizeof(local)) != 0) {
        close(fd);
        goto fail;
    }
    if (modify_qp_rtr(h.qp, &remote, gid_index) != 0 || modify_qp_rts(h.qp, local.psn) != 0) {
        close(fd);
        goto fail;
    }

    struct ibv_sge sge;
    memset(&sge, 0, sizeof(sge));
    sge.addr = (uint64_t)(uintptr_t)local_addr;
    sge.length = (uint32_t)size;
    sge.lkey = h.mr->lkey;
    struct ibv_send_wr wr;
    struct ibv_send_wr *bad = NULL;
    memset(&wr, 0, sizeof(wr));
    wr.wr_id = 1;
    wr.sg_list = &sge;
    wr.num_sge = 1;
    wr.opcode = IBV_WR_RDMA_WRITE;
    wr.send_flags = IBV_SEND_SIGNALED;
    wr.wr.rdma.remote_addr = remote.addr;
    wr.wr.rdma.rkey = remote.rkey;
    if (size > remote.size || ibv_post_send(h.qp, &wr, &bad) != 0) {
        close(fd);
        goto fail;
    }
    struct ibv_wc wc;
    int polls = 0;
    do {
        int n = ibv_poll_cq(h.cq, 1, &wc);
        if (n < 0) {
            close(fd);
            goto fail;
        }
        if (n > 0) {
            break;
        }
        usleep(1000);
    } while (++polls < 15000);
    if (polls >= 15000 || wc.status != IBV_WC_SUCCESS) {
        close(fd);
        goto fail;
    }
    char done = 'D';
    (void)send_all(fd, &done, 1);
    close(fd);
    if (h.mr != NULL) ibv_dereg_mr(h.mr);
    if (h.qp != NULL) ibv_destroy_qp(h.qp);
    if (h.cq != NULL) ibv_destroy_cq(h.cq);
    if (h.pd != NULL) ibv_dealloc_pd(h.pd);
    if (h.ctx != NULL) ibv_close_device(h.ctx);
    return 0;

fail:
    if (h.mr != NULL) ibv_dereg_mr(h.mr);
    if (h.qp != NULL) ibv_destroy_qp(h.qp);
    if (h.cq != NULL) ibv_destroy_cq(h.cq);
    if (h.pd != NULL) ibv_dealloc_pd(h.pd);
    if (h.ctx != NULL) ibv_close_device(h.ctx);
    return errno ? -errno : -1;
}
