from asynctest import TestCase as AsyncTestCase

import json

from ...storage.error import StorageNotFoundError
from ..util import bytes_to_b58
from ..key_type import ED25519
from ...core.in_memory import InMemoryProfile
from ...storage.in_memory import InMemoryStorage
from ..key_pair import KeyPairStorageManager, KEY_PAIR_STORAGE_TYPE

import aries_cloudagent.askar.didcomm.v1 as test_module
from aries_askar import AskarError, Key, KeyAlg, Session
import pytest
import aries_cloudagent.askar.profile

@pytest.fixture()
async def session():
    print("###1###")
    context = InjectionContext()
    profile = await profile.AskarProfileManager().provision(
        context,
        {
            "name": ":memory:",
            "key": await profile.AskarProfileManager.generate_store_key(),
            "key_derivation_method": "RAW",  # much faster than using argon-hashed keys
        },
    )
    print("###2###")
    async with profile.session() as session:
        print("TEST SESSION")
        yield session.handle
    del session
    await profile.close()

class TestKeyPairStorageManager(AsyncTestCase):
    test_public_key = b"somepublickeybytes"
    test_secret = b"verysecretkey"

    async def setUp(self):
        self.profile = InMemoryProfile.test_profile()
        self.store = InMemoryStorage(self.profile)
        self.key_pair_mgr = KeyPairStorageManager(self.store)

    @pytest.mark.asyncio
    async def test_create_key_pair(self, session: Session):
        #await self.key_pair_mgr.store_key_pair(
        #    public_key=self.test_public_key,
        #    secret_key=self.test_secret,
        #    key_type=ED25519,
        #)

        alg = KeyAlg.ED25519
        secret = "00000000000000000000000Recipient" # Wie im C# Test
        test_key = Key.from_secret_bytes(alg, secret)
        verkey_bytes = test_key.get_public_bytes()

        verkey = bytes_to_b58(verkey_bytes)

        print(verkey)

        #key_pair = await self.key_pair_mgr.get_key_pair(verkey)
        
        message_unknown_alg = json.dumps(
            {
                "protected": "eyJlbmMiOiJ4Y2hhY2hhMjBwb2x5MTMwNV9pZXRmIiwidHlwIjoiSldNLzEuMCIsImFsZyI6IkFub25jcnlwdCIsInJlY2lwaWVudHMiOlt7ImVuY3J5cHRlZF9rZXkiOiJkVWlZRVdxTHk1eUc4T0VpYTB5UFNtSEhIMUVmUEk1VWQtQ1diY1RKMFNONWJSQ0RZNmNuTm1RZWRxRkZvQjhQTHJvR3RoellKVnBOSGtmNjdZNTdCVGRjTkRfa09ueDN2ZEdIYTQybFBtMD0iLCJoZWFkZXIiOnsia2lkIjoiR1VpNFc2cVFDU0JhWVRCcEU4dlRXRFNTc3dTWUN3aTFkYURKZ0d5TXpFd1kifX1dfQ==",#b64url(json.dumps({"alg": "NOT-SUPPORTED"})),
                #"recipients": [{"header": {"kid": "bob"}, "encrypted_key": "MTIzNA"}],
                "iv": "ch4OOmVAgCvkC0Fz",
                "ciphertext": "MTIzNA",#"MngxC_XHNc02kF1j4_E4OvLesy_T8xnn5NyRjtOaEpjgCbIPC6EmAmbxX7_wNEBNRAiYkrZyXVJFAkZGgwRBQAi6aZGI5VYkIsNDd1MMsuGdxyhtXCQc3ub2ZGqlb07tBP4DWuCrIWX4WLNKMHetQqnsn82mN8sSutleV5fCr4MeIkq0GHGbge2obnKpZSqYfSOW3tR-a1iTqtUZP2KLc-CgofeXZgcyBoEzQZihg-k8xsC1_r4FMIhDI2jlU4pPU1xk8_y4_ghiIT79JIUr_vjTmtIhtA==",
                "tag": "1_h7oMf2SXv7N0UssXB7cA==",
            }
        )

        _ = await test_module.unpack_message(session, message_unknown_alg)
        #record = await self.store.find_record(KEY_PAIR_STORAGE_TYPE, {"verkey": verkey})

        #assert record

        #value = json.loads(record.value)

        #assert record.tags == {"verkey": verkey, "key_type": ED25519.key_type}
        #assert value["verkey"] == verkey
        #assert value["secret_key"] == bytes_to_b58(self.test_secret)
        #assert value["metadata"] == {}
        #assert value["key_type"] == ED25519.key_type

    async def test_get_key_pair(self):
        await self.key_pair_mgr.store_key_pair(
            public_key=self.test_public_key,
            secret_key=self.test_secret,
            key_type=ED25519,
        )

        verkey = bytes_to_b58(self.test_public_key)

        key_pair = await self.key_pair_mgr.get_key_pair(verkey)

        assert key_pair["verkey"] == verkey
        assert key_pair["secret_key"] == bytes_to_b58(self.test_secret)
        assert key_pair["metadata"] == {}
        assert key_pair["key_type"] == ED25519.key_type

    async def test_get_key_pair_x_not_found(self):
        with self.assertRaises(StorageNotFoundError):
            await self.key_pair_mgr.get_key_pair("not_existing_verkey")

    async def test_delete_key_pair(self):
        await self.key_pair_mgr.store_key_pair(
            public_key=self.test_public_key,
            secret_key=self.test_secret,
            key_type=ED25519,
        )

        verkey = bytes_to_b58(self.test_public_key)

        record = await self.store.find_record(KEY_PAIR_STORAGE_TYPE, {"verkey": verkey})
        assert record

        await self.key_pair_mgr.delete_key_pair(verkey)

        # should be deleted now
        with self.assertRaises(StorageNotFoundError):
            await self.key_pair_mgr.delete_key_pair(verkey)

    async def test_delete_key_pair_x_not_found(self):
        with self.assertRaises(StorageNotFoundError):
            await self.key_pair_mgr.delete_key_pair("non_existing_verkey")

    async def test_update_key_pair_metadata(self):
        await self.key_pair_mgr.store_key_pair(
            public_key=self.test_public_key,
            secret_key=self.test_secret,
            key_type=ED25519,
            metadata={"some": "data"},
        )

        verkey = bytes_to_b58(self.test_public_key)

        record = await self.store.find_record(KEY_PAIR_STORAGE_TYPE, {"verkey": verkey})
        assert record
        value = json.loads(record.value)

        assert value["metadata"] == {"some": "data"}

        await self.key_pair_mgr.update_key_pair_metadata(verkey, {"some_other": "data"})

        record = await self.store.find_record(KEY_PAIR_STORAGE_TYPE, {"verkey": verkey})
        assert record
        value = json.loads(record.value)

        assert value["metadata"] == {"some_other": "data"}

    async def test_update_key_pair_metadata_x_not_found(self):
        with self.assertRaises(StorageNotFoundError):
            await self.key_pair_mgr.update_key_pair_metadata("non_existing_verkey", {})
