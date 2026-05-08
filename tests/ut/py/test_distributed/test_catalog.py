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


def test_catalog_version_mismatch():
    catalog = Catalog()
    cid, version = catalog.register(lambda args: None)
    payload = catalog.export_payload(cid, version)
    with pytest.raises(CatalogError, match="version mismatch"):
        catalog.install_from_payload(cid, version + 1, payload)
