import json

from asynctest import mock as async_mock
import pytest

from aries_askar import AskarError, Key, KeyAlg, Session

from ....config.injection_context import InjectionContext
from ....utils.jwe import JweRecipient, b64url, JweEnvelope

from ...profile import AskarProfileManager
from .. import v2 as test_module

from .. import v1 as test_moduleV1
from aries_cloudagent.wallet.util import bytes_to_b58

ALICE_KID = "did:example:alice#key-1"
BOB_KID = "did:example:bob#key-1"
CAROL_KID = "did:example:carol#key-2"
MESSAGE = b"Expecto patronum"


@pytest.fixture()
async def session():
    context = InjectionContext()
    profile = await AskarProfileManager().provision(
        context,
        {
            "name": ":memory:",
            "key": await AskarProfileManager.generate_store_key(),
            "key_derivation_method": "RAW",  # much faster than using argon-hashed keys
        },
    )
    async with profile.session() as session:
        yield session.handle
    del session
    await profile.close()


@pytest.mark.askar
class TestAskarDidCommV2:
    @pytest.mark.asyncio
    async def test_es_round_trip(self, session: Session):
        alg = KeyAlg.X25519
        bob_sk = Key.generate(alg)
        bob_pk = Key.from_jwk(bob_sk.get_jwk_public())
        carol_sk = Key.generate(KeyAlg.P256)  # testing mixed recipient key types
        carol_pk = Key.from_jwk(carol_sk.get_jwk_public())

        enc_message = test_module.ecdh_es_encrypt(
            {BOB_KID: bob_pk, CAROL_KID: carol_pk}, MESSAGE
        )

        # receiver must have the private keypair accessible
        await session.insert_key("my_sk", bob_sk, tags={"kid": BOB_KID})

        plaintext, recip_kid, sender_kid = await test_module.unpack_message(
            session, enc_message
        )
        assert recip_kid == BOB_KID
        assert sender_kid is None
        assert plaintext == MESSAGE

    @pytest.mark.asyncio
    async def test_es_encrypt_x(self, session: Session):
        alg = KeyAlg.X25519
        bob_sk = Key.generate(alg)
        bob_pk = Key.from_jwk(bob_sk.get_jwk_public())

        with pytest.raises(
            test_module.DidcommEnvelopeError, match="No message recipients"
        ):
            _ = test_module.ecdh_es_encrypt({}, MESSAGE)

        with async_mock.patch(
            "aries_askar.Key.generate",
            async_mock.MagicMock(side_effect=AskarError(99, "")),
        ):
            with pytest.raises(
                test_module.DidcommEnvelopeError,
                match="Error creating content encryption key",
            ):
                _ = test_module.ecdh_es_encrypt({BOB_KID: bob_pk}, MESSAGE)

        with async_mock.patch(
            "aries_askar.Key.aead_encrypt",
            async_mock.MagicMock(side_effect=AskarError(99, "")),
        ):
            with pytest.raises(
                test_module.DidcommEnvelopeError,
                match="Error encrypting",
            ):
                _ = test_module.ecdh_es_encrypt({BOB_KID: bob_pk}, MESSAGE)

    @pytest.mark.asyncio
    async def test_es_decrypt_x(self):
        alg = KeyAlg.X25519
        bob_sk = Key.generate(alg)

        message_unknown_alg = JweEnvelope(
            protected={"alg": "NOT-SUPPORTED"},
        )
        message_unknown_alg.add_recipient(
            JweRecipient(encrypted_key=b"0000", header={"kid": BOB_KID})
        )
        with pytest.raises(
            test_module.DidcommEnvelopeError,
            match="Unsupported ECDH-ES algorithm",
        ):
            _ = test_module.ecdh_es_decrypt(
                message_unknown_alg,
                BOB_KID,
                bob_sk,
            )

        message_unknown_enc = JweEnvelope(
            protected={"alg": "ECDH-ES+A128KW", "enc": "UNKNOWN"},
        )
        message_unknown_enc.add_recipient(
            JweRecipient(encrypted_key=b"0000", header={"kid": BOB_KID})
        )
        with pytest.raises(
            test_module.DidcommEnvelopeError,
            match="Unsupported ECDH-ES content encryption",
        ):
            _ = test_module.ecdh_es_decrypt(
                message_unknown_enc,
                BOB_KID,
                bob_sk,
            )

        message_invalid_epk = JweEnvelope(
            protected={"alg": "ECDH-ES+A128KW", "enc": "A256GCM", "epk": {}},
        )
        message_invalid_epk.add_recipient(
            JweRecipient(encrypted_key=b"0000", header={"kid": BOB_KID})
        )
        with pytest.raises(
            test_module.DidcommEnvelopeError,
            match="Error loading ephemeral key",
        ):
            _ = test_module.ecdh_es_decrypt(
                message_invalid_epk,
                BOB_KID,
                bob_sk,
            )

    @pytest.mark.asyncio
    async def test_1pu_round_trip(self, session: Session):
        alg = KeyAlg.X25519
        alice_sk = Key.generate(alg)
        alice_pk = Key.from_jwk(alice_sk.get_jwk_public())
        bob_sk = Key.generate(alg)
        bob_pk = Key.from_jwk(bob_sk.get_jwk_public())

        enc_message = test_module.ecdh_1pu_encrypt(
            {BOB_KID: bob_pk}, ALICE_KID, alice_sk, MESSAGE
        )

        # receiver must have the private keypair accessible
        await session.insert_key("my_sk", bob_sk, tags={"kid": BOB_KID})
        # for now at least, insert the sender public key so it can be resolved
        await session.insert_key("alice_pk", alice_pk, tags={"kid": ALICE_KID})

        plaintext, recip_kid, sender_kid = await test_module.unpack_message(
            session, enc_message
        )
        assert recip_kid == BOB_KID
        assert sender_kid == ALICE_KID
        assert plaintext == MESSAGE

    @pytest.mark.asyncio
    async def test_1pu_encrypt_x(self, session: Session):
        alg = KeyAlg.X25519
        alice_sk = Key.generate(alg)
        bob_sk = Key.generate(alg)
        bob_pk = Key.from_jwk(bob_sk.get_jwk_public())

        with pytest.raises(
            test_module.DidcommEnvelopeError, match="No message recipients"
        ):
            _ = test_module.ecdh_1pu_encrypt({}, ALICE_KID, alice_sk, MESSAGE)

        alt_sk = Key.generate(KeyAlg.P256)
        alt_pk = Key.from_jwk(alt_sk.get_jwk_public())
        with pytest.raises(
            test_module.DidcommEnvelopeError, match="key types must be consistent"
        ):
            _ = test_module.ecdh_1pu_encrypt(
                {BOB_KID: bob_pk, "alt": alt_pk}, ALICE_KID, alice_sk, MESSAGE
            )

        with async_mock.patch(
            "aries_askar.Key.generate",
            async_mock.MagicMock(side_effect=AskarError(99, "")),
        ):
            with pytest.raises(
                test_module.DidcommEnvelopeError,
                match="Error creating content encryption key",
            ):
                _ = test_module.ecdh_1pu_encrypt(
                    {BOB_KID: bob_pk}, ALICE_KID, alice_sk, MESSAGE
                )

        with async_mock.patch(
            "aries_askar.Key.aead_encrypt",
            async_mock.MagicMock(side_effect=AskarError(99, "")),
        ):
            with pytest.raises(
                test_module.DidcommEnvelopeError,
                match="Error encrypting",
            ):
                _ = test_module.ecdh_1pu_encrypt(
                    {BOB_KID: bob_pk}, ALICE_KID, alice_sk, MESSAGE
                )

    @pytest.mark.asyncio
    async def test_1pu_decrypt_x(self):
        alg = KeyAlg.X25519
        alice_sk = Key.generate(alg)
        alice_pk = Key.from_jwk(alice_sk.get_jwk_public())
        bob_sk = Key.generate(alg)

        message_unknown_alg = JweEnvelope(
            protected={"alg": "NOT-SUPPORTED"},
        )
        message_unknown_alg.add_recipient(
            JweRecipient(encrypted_key=b"0000", header={"kid": BOB_KID})
        )
        with pytest.raises(
            test_module.DidcommEnvelopeError,
            match="Unsupported ECDH-1PU algorithm",
        ):
            _ = test_module.ecdh_1pu_decrypt(
                message_unknown_alg,
                BOB_KID,
                bob_sk,
                alice_pk,
            )

        message_unknown_enc = JweEnvelope(
            protected={"alg": "ECDH-1PU+A128KW", "enc": "UNKNOWN"},
        )
        message_unknown_enc.add_recipient(
            JweRecipient(encrypted_key=b"0000", header={"kid": BOB_KID})
        )
        with pytest.raises(
            test_module.DidcommEnvelopeError,
            match="Unsupported ECDH-1PU content encryption",
        ):
            _ = test_module.ecdh_1pu_decrypt(
                message_unknown_enc, BOB_KID, bob_sk, alice_pk
            )

        message_invalid_epk = JweEnvelope(
            protected={"alg": "ECDH-1PU+A128KW", "enc": "A256CBC-HS512", "epk": {}},
        )
        message_invalid_epk.add_recipient(
            JweRecipient(encrypted_key=b"0000", header={"kid": BOB_KID})
        )
        with pytest.raises(
            test_module.DidcommEnvelopeError,
            match="Error loading ephemeral key",
        ):
            _ = test_module.ecdh_1pu_decrypt(
                message_invalid_epk,
                BOB_KID,
                bob_sk,
                alice_pk,
            )
    
    @pytest.mark.asyncio
    async def test_unpack_message_any_x(self, session: Session):
        
        #Init Verkey, Did and store them in wallet as in our PackUnpack Tests (MessageServiceTestsV1V2") in Aries
        alg = KeyAlg.ED25519
        secret = "00000000000000000000000Recipient" 
        test_key = Key.from_secret_bytes(alg, secret)

        verkey_bytes = test_key.get_public_bytes()
        seckey_bytes = test_key.get_secret_bytes()

        verkey = bytes_to_b58(verkey_bytes)
        seckey = bytes_to_b58(seckey_bytes)
        did = bytes_to_b58(verkey_bytes[0:16])
        print("Recipient Verkey as in C#: "+ verkey)
        print("Recipient Secretkey as in C#: "+ seckey)
        print("Recipient DID as in C#: "+ did)
        
        #todo - Save keyPair with Id verkey in the wallet --> Look for code examples in this project
        await session.insert_key(verkey, test_key)
        entry = await session.fetch_key(verkey)
        #print("Key added and directly read publicbytes:" + bytes_to_b58(entry.key.get_public_bytes()))
        #print("Key added and directly read secretbytes:" + bytes_to_b58(entry.key.get_secret_bytes()))
        message_V1_pack = json.dumps(
            {
                "protected": "eyJlbmMiOiJ4Y2hhY2hhMjBwb2x5MTMwNV9pZXRmIiwidHlwIjoiSldNLzEuMCIsImFsZyI6IkFub25jcnlwdCIsInJlY2lwaWVudHMiOlt7ImVuY3J5cHRlZF9rZXkiOiJTd3dPWEE4NWl5XzhlazhyWmh0RDUwaWFUdHJBWmFoVkhkMmJoamRCcEF3Sm9SUDloZVg0cVI1ZmozQU44ZnhvMEl5ZlhSTjZrM0pVVFRhWVlfb1ctbXVtT2ZjSWhmbWs3MU9oX2xJeks1OD0iLCJoZWFkZXIiOnsia2lkIjoiR1VpNFc2cVFDU0JhWVRCcEU4dlRXRFNTc3dTWUN3aTFkYURKZ0d5TXpFd1kifX1dfQ==",#b64url(json.dumps({"alg": "NOT-SUPPORTED"})),
                #"recipients": [{"header": {"kid": "bob"}, "encrypted_key": "MTIzNA"}],
                "iv": "7SFfQv7JkUxw-IfB",
                "ciphertext": "kjNxIuTjRXmOWhOzlF53R1G3sh81tsK2vQOXH8AQS23JPWXh1WA0z2w0FsSi7aqzt0ssbDsj-Y6YzOq5mQkijCaiQBN9o4kr95oOEjnFOzEwsYp3-bnjAtZWp_JeAje_oHt90Ia5zgYSdiq9QHwOBjAIPCxr-bo2TkNDj-HMYXo4z4bXFtKXa86eQ2AAFtWqbAukK8ZFGpugbGXlXThi4afY70zGpyBz0Ol6q880qk5fx0CWf8ig20tADcXvVbNcQ88xcDhi6AzMngrytkPVyA9gvhJfNQ==",#"MTIzNA",
                "tag": "o1YFpsc0p5h-Ae-AUfUC7g==",
            }
        )

        message_V2_pack = json.dumps(
            {
                "protected": "ueyJlbmMiOiJ4Y2hhY2hhMjBwb2x5MTMwNV9pZXRmIiwidHlwIjoiSldNLzEuMCIsImFsZyI6IkFub25jcnlwdCIsInJlY2lwaWVudHMiOlt7ImVuY3J5cHRlZF9rZXkiOiJ1OHktM2RUaFB5OVRpQTNySDdrVTRFLWtabXIyUElXVjlRemxvVGpOOHBTd3k2dmQ0LTE5V0tEaHdDSGNNa3JxeDluXzctUkdUdU53N2llazhaN1NKczlGaUU4NDZVMjBQa3hUTnd4azhoVjY0bGpfb2lzZUFjX1JQbGxRIiwiaGVhZGVyIjp7ImtpZCI6IjcxamVoNXdGTmNIZGJiYnFtMTVNaXRtaEpCWnF1UTluOWRkWXFiMlBiOHJtIiwic2VuZGVyIjpudWxsLCJpdiI6bnVsbH19XX0",
                #"recipients": [{"header": {"kid": "bob"}, "encrypted_key": "MTIzNA"}],
                "iv": "1ZcHTdMCxWHXYRg3b8wdjZQseqiw9W+2",
                "ciphertext": "0AI7wN3/E91+OGgWjE0fUnYociiaos0QWnTHAFVPGgbid8SiNoFOdtkTfV7s0HnAoMbVz+jy6WW6fPDXBeByEsVDHlOGKN06Yp0OdwNO7DUUKqHSnU4AvcCD0tn9wBPFienlN7c8W6kmQzCfOt3QzQsYFSxRpDfcs581vhPoFq05L/1S1PXVDz9BO8uBYYoAq1Vz9EzLvpyhz+O+OlY4OCROnqc+F4bKEIuKRtzwNCnQsJUjNS59gPIbBP9gm1tAMwL57GizGql3fkWVIloL4gOWlQ3dFg==",
                "tag": "+B0zW8/XL2uBN8WqgOjNTA==",
            }
        )

        _ = await test_moduleV1.unpack_message(session, message_V2_pack)

    @pytest.mark.asyncio
    async def test_unpack_message_1pu_x(self, session: Session):
        alg = KeyAlg.X25519
        alice_sk = Key.generate(alg)
        alice_pk = Key.from_jwk(alice_sk.get_jwk_public())
        bob_sk = Key.generate(alg)
        bob_pk = Key.from_jwk(bob_sk.get_jwk_public())

        # receiver must have the private keypair accessible
        await session.insert_key("my_sk", bob_sk, tags={"kid": BOB_KID})
        # for now at least, insert the sender public key so it can be resolved
        await session.insert_key("alice_pk", alice_pk, tags={"kid": ALICE_KID})

        message_1pu_no_skid = json.dumps(
            {
                "protected": b64url(json.dumps({"alg": "ECDH-1PU+A128KW"})),
                "recipients": [{"header": {"kid": BOB_KID}, "encrypted_key": "MTIzNA"}],
                "iv": "MTIzNA",
                "ciphertext": "MTIzNA",
                "tag": "MTIzNA",
            }
        )

        with pytest.raises(
            test_module.DidcommEnvelopeError,
            match="Sender key ID not provided",
        ):
            _ = await test_module.unpack_message(session, message_1pu_no_skid)

        message_1pu_unknown_skid = json.dumps(
            {
                "protected": b64url(
                    json.dumps({"alg": "ECDH-1PU+A128KW", "skid": "UNKNOWN"})
                ),
                "recipients": [{"header": {"kid": BOB_KID}, "encrypted_key": "MTIzNA"}],
                "iv": "MTIzNA",
                "ciphertext": "MTIzNA",
                "tag": "MTIzNA",
            }
        )

        with pytest.raises(
            test_module.DidcommEnvelopeError,
            match="Sender public key not found",
        ):
            _ = await test_module.unpack_message(session, message_1pu_unknown_skid)

        message_1pu_apu_invalid = json.dumps(
            {
                "protected": b64url(
                    json.dumps({"alg": "ECDH-1PU+A128KW", "skid": "A", "apu": "A"})
                ),
                "recipients": [{"header": {"kid": BOB_KID}, "encrypted_key": "MTIzNA"}],
                "iv": "MTIzNA",
                "ciphertext": "MTIzNA",
                "tag": "MTIzNA",
            }
        )

        with pytest.raises(
            test_module.DidcommEnvelopeError,
            match="Invalid apu value",
        ):
            _ = await test_module.unpack_message(session, message_1pu_apu_invalid)

        message_1pu_apu_mismatch = json.dumps(
            {
                "protected": b64url(
                    json.dumps(
                        {
                            "alg": "ECDH-1PU+A128KW",
                            "skid": ALICE_KID,
                            "apu": b64url("UNKNOWN"),
                        }
                    )
                ),
                "recipients": [{"header": {"kid": BOB_KID}, "encrypted_key": "MTIzNA"}],
                "iv": "MTIzNA",
                "ciphertext": "MTIzNA",
                "tag": "MTIzNA",
            }
        )

        with pytest.raises(
            test_module.DidcommEnvelopeError,
            match="Mismatch between skid and apu",
        ):
            _ = await test_module.unpack_message(session, message_1pu_apu_mismatch)
