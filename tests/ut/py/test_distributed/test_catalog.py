import pytest

from simpler.distributed.catalog import Catalog, CatalogError


def test_catalog_export_install_lookup():
    l4 = Catalog()

    def fn(args):
        return args.scalar(0) + 1

    cid, version = l4.register(fn)
    payload = l4.export_payload(cid, version)

    l3 = Catalog()
    l3.install_from_payload(cid, version, payload)

    got = l3.lookup(cid, version)
    assert got is not None


def test_catalog_pull_mock_install():
    l4 = Catalog()
    cid, version = l4.register(lambda args: args.scalar(0) * 2)

    class MockClient:
        def call_unary(self, method, req, timeout=None):
            assert method == "Catalog.PullCallable"
            return type("Payload", (), {"callable_id": cid, "version": version, "pickled": l4.export_payload(cid, version)})

    l3 = Catalog()
    req = type("Req", (), {"callable_id": cid, "version": version})()
    payload = MockClient().call_unary("Catalog.PullCallable", req)
    l3.install_from_payload(payload.callable_id, payload.version, payload.pickled)

    assert l3.lookup(cid, version) is not None


def test_catalog_version_mismatch():
    catalog = Catalog()
    cid, version = catalog.register(lambda args: None)
    payload = catalog.export_payload(cid, version)
    with pytest.raises(CatalogError, match="version mismatch"):
        catalog.install_from_payload(cid, version + 1, payload)
