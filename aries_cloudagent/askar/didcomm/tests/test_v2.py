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
        print(" ")
        print("---------------- Create and save Wallet keys for test --------------------")
        print(" ")
        alg = KeyAlg.ED25519
        secretTheirVerkey = "00000000000000000000000000Sender"
        test_keyRecipient = Key.from_secret_bytes(alg, secretTheirVerkey) 

        verkey_recipient_bytes = test_keyRecipient.get_public_bytes()
        seckey_recipient_bytes = test_keyRecipient.get_secret_bytes()

        verkey_recipient = bytes_to_b58(verkey_recipient_bytes)
        seckey_recipient = bytes_to_b58(seckey_recipient_bytes)
        did_recipient = bytes_to_b58(verkey_recipient_bytes[0:16])
        print("Recipient Verkey as in C#: "+ verkey_recipient)
        print("Recipient Secretkey as in C#: "+ seckey_recipient)
        print("Recipient DID as in C#: "+ did_recipient)

        secretMyVerkey = "00000000000000000000000Recipient" 
        test_keySender = Key.from_secret_bytes(alg, secretMyVerkey)

        verkey_sender_bytes = test_keySender.get_public_bytes()
        seckey_sender_bytes = test_keySender.get_secret_bytes()

        verkey_sender = bytes_to_b58(verkey_sender_bytes)
        seckey_sender = bytes_to_b58(seckey_sender_bytes)
        did_sender = bytes_to_b58(verkey_sender_bytes[0:16])
        print("Sender Verkey as in C#: "+ verkey_sender)
        print("Sender Secretkey as in C#: "+ seckey_sender)
        print("Sender DID as in C#: "+ did_sender)
        
        #Insert Keys in wallet
        await session.insert_key(verkey_recipient, test_keyRecipient)
        await session.insert_key(verkey_sender, test_keySender)

        message_V1Service_pack_anon = json.dumps(
            {
                "protected": "eyJlbmMiOiJ4Y2hhY2hhMjBwb2x5MTMwNV9pZXRmIiwidHlwIjoiSldNLzEuMCIsImFsZyI6IkFub25jcnlwdCIsInJlY2lwaWVudHMiOlt7ImVuY3J5cHRlZF9rZXkiOiJTd3dPWEE4NWl5XzhlazhyWmh0RDUwaWFUdHJBWmFoVkhkMmJoamRCcEF3Sm9SUDloZVg0cVI1ZmozQU44ZnhvMEl5ZlhSTjZrM0pVVFRhWVlfb1ctbXVtT2ZjSWhmbWs3MU9oX2xJeks1OD0iLCJoZWFkZXIiOnsia2lkIjoiR1VpNFc2cVFDU0JhWVRCcEU4dlRXRFNTc3dTWUN3aTFkYURKZ0d5TXpFd1kifX1dfQ==",#b64url(json.dumps({"alg": "NOT-SUPPORTED"})),
                #"recipients": [{"header": {"kid": "bob"}, "encrypted_key": "MTIzNA"}],
                "iv": "7SFfQv7JkUxw-IfB",
                "ciphertext": "kjNxIuTjRXmOWhOzlF53R1G3sh81tsK2vQOXH8AQS23JPWXh1WA0z2w0FsSi7aqzt0ssbDsj-Y6YzOq5mQkijCaiQBN9o4kr95oOEjnFOzEwsYp3-bnjAtZWp_JeAje_oHt90Ia5zgYSdiq9QHwOBjAIPCxr-bo2TkNDj-HMYXo4z4bXFtKXa86eQ2AAFtWqbAukK8ZFGpugbGXlXThi4afY70zGpyBz0Ol6q880qk5fx0CWf8ig20tADcXvVbNcQ88xcDhi6AzMngrytkPVyA9gvhJfNQ==",#"MTIzNA",
                "tag": "o1YFpsc0p5h-Ae-AUfUC7g==",
            }
        )

        message_V2Service_pack_anon = json.dumps(
            {
                "protected": "eyJlbmMiOiJjaGFjaGEyMHBvbHkxMzA1X2lldGYiLCJ0eXAiOiJKV00vMS4wIiwiYWxnIjoiQW5vbmNyeXB0IiwicmVjaXBpZW50cyI6W3siZW5jcnlwdGVkX2tleSI6IjhJRTZyOVpaMFlFMmFobW5XeXFxM0JsQzQ2eVJOWXEzYmpFUUROd2JrMnU5TmdqOUpjUHlSU1dBNVlMUEpfTmE3dDNEVlBkX2lENnJfbzdLM0kwMTFVS2dpWjVDdmpDNVhXOWMzRVJtS0dZPSIsImhlYWRlciI6eyJraWQiOiJHVWk0VzZxUUNTQmFZVEJwRTh2VFdEU1Nzd1NZQ3dpMWRhREpnR3lNekV3WSIsInNlbmRlciI6bnVsbCwiaXYiOm51bGx9fV19",
                "iv": "DZJ7_6FZgF_dP_if",
                "ciphertext": "oR5datlR75gimD4MXHbuc7DvzgtSBFfVB4YFfIoxhQKYj2V0tlARL7VvMX_LjLLTCGSFtyKHbTaOF5HljSXxqo5F0wMo5zLxE17bkt6ZzPX5JTLR4TDivz5086KMhMEPRMWxM7D3ksfP9wHCGK7jZVgJaDBezs_Mn3D1QN7dc1bYseuJy89bceSZBz-GuGst7l-ill0yK36aikFOfQqmbPJ_SmVL6-adchZYu0oD68jDfPzJ0Vsta-4LNbkQDLapYDyFj40m4EW-coqUixFC5kEJFs9rmQ==",
                "tag": "iVly_gYW4oMGy3flGY4T5w==",
            }
        )

        message_V2Service_pack_auth = json.dumps(
            {
                "protected": "eyJlbmMiOiJjaGFjaGEyMHBvbHkxMzA1X2lldGYiLCJ0eXAiOiJKV00vMS4wIiwiYWxnIjoiQXV0aGNyeXB0IiwicmVjaXBpZW50cyI6W3siZW5jcnlwdGVkX2tleSI6ImhsSk1QZVpWX2dxdFpWS0UxaGgyVTFodnVBTUNkSDRyZV9IakZrOE1SS0Z2QVZvYWRGN09wMXlwQnQzMFhObGwiLCJoZWFkZXIiOnsia2lkIjoiR0dtWjR1NjYyaVpBeU10VVlwWDRBaWF3TVREZERvNFdTeUJOWEpFN0FDQ2UiLCJzZW5kZXIiOiJHRklFaUJmNGd3OUZnZmJYTWhrVlNROW52ZXF3QlFvSlZzZlhnN3lwSTJyYmJicUxHUi1tRS1hS0xGWjd5Ump4NFI0NDd5S2xZc28wM2lYRTJsc1diUUxOZmROMFRYc29KZ0hsSmNOcy02VFkxR09uTGE0eWNVZWRoaFk9IiwiaXYiOiJsellMU2R1SlI2R1pMQWs5MmRYa1hhRHE3ZFRMTTFOMCJ9fV19",
                #"recipients": [{"header": {"kid": "bob"}, "encrypted_key": "MTIzNA"}],
                "iv": "5Ig9DeGxGP1D9bE0",
                "ciphertext": "AxyEu7k9PSEbKkBkbcJWXHv_WCQ7CUcciLp5MJTSMM4B4zHiAuNFIvXc7dpwLj0ZQGA2F4me9ebg5N8cZXczczHxbAoQmWWyL6n7Gb0Wq7siCZrjIrTU1hWFaerqJUxYqXxpszhAfMD8G5VRdqunhOObtYuJ6laeF44Nsw8rBtps67V62yvzYT6VmuamlhOCY6G4ZgQFtWEwXTCA0NLn70goQG5levY50czkqskZwSb0fqAp9d4a0qxeXD06IRhsog1up1fPEuNVbtMxrIiBfiDTN9R71g==",
                "tag": "B92oMQ1O4T12W70nvI1A3g==",
            }
        )

        demoList = [123,34,112,114,111,116,101,99,116,101,100,34,58,34,101,121,74,108,98,109,77,105,79,105,74,52,89,50,104,104,89,50,104,104,77,106,66,119,98,50,120,53,77,84,77,119,78,86,57,112,90,88,82,109,73,105,119,105,100,72,108,119,73,106,111,105,83,108,100,78,76,122,69,117,77,67,73,115,73,109,70,115,90,121,73,54,73,107,70,117,98,50,53,106,99,110,108,119,100,67,73,115,73,110,74,108,89,50,108,119,97,87,86,117,100,72,77,105,79,108,116,55,73,109,86,117,89,51,74,53,99,72,82,108,90,70,57,114,90,88,107,105,79,105,74,108,99,51,70,76,90,109,120,104,82,85,86,72,85,50,53,105,98,48,86,85,97,50,104,77,84,69,108,79,76,48,49,72,89,107,82,122,81,48,104,87,89,86,99,51,98,72,90,90,81,87,82,117,78,70,82,108,83,69,57,114,90,85,77,119,89,106,104,76,98,84,82,119,97,48,104,48,79,85,57,110,82,87,78,86,100,69,116,113,89,108,108,66,97,70,112,54,78,50,108,120,78,86,99,114,99,109,100,85,100,107,116,71,100,83,116,74,101,84,74,82,97,70,74,86,90,108,108,120,81,110,108,68,83,50,104,53,83,69,90,107,100,122,48,105,76,67,74,111,90,87,70,107,90,88,73,105,79,110,115,105,97,50,108,107,73,106,111,105,82,49,86,112,78,70,99,50,99,86,70,68,85,48,74,104,87,86,82,67,99,69,85,52,100,108,82,88,82,70,78,84,99,51,100,84,87,85,78,51,97,84,70,107,89,85,82,75,90,48,100,53,84,88,112,70,100,49,107,105,76,67,74,122,90,87,53,107,90,88,73,105,79,109,53,49,98,71,119,115,73,109,108,50,73,106,112,117,100,87,120,115,102,88,49,100,102,81,61,61,34,44,34,105,118,34,58,34,79,51,117,120,54,119,99,50,122,77,99,73,68,117,84,87,34,44,34,99,105,112,104,101,114,116,101,120,116,34,58,34,50,80,97,47,81,65,87,66,88,50,52,110,83,84,102,104,66,86,76,104,114,109,86,54,69,74,104,122,108,71,71,84,87,66,83,84,104,71,108,77,99,48,89,116,71,122,118,110,121,81,67,86,55,117,65,116,110,78,48,81,66,67,104,75,100,50,53,114,56,68,106,119,111,50,104,69,51,54,122,80,56,76,121,82,69,98,69,84,87,120,121,48,80,65,53,81,98,99,52,50,48,101,101,68,112,75,84,48,77,109,108,84,115,114,67,113,108,89,75,69,88,89,47,79,65,65,43,113,90,52,116,89,50,86,55,53,68,72,121,109,109,52,73,69,85,68,103,70,67,100,101,50,70,74,83,89,85,107,71,70,117,71,87,115,82,110,109,56,47,105,77,100,114,74,98,89,85,107,117,114,108,83,115,102,105,115,67,69,105,81,81,82,51,69,106,57,43,111,86,74,101,110,80,75,102,101,47,73,65,76,98,101,81,71,72,76,112,122,49,107,98,53,77,110,121,48,77,66,71,103,89,113,65,52,101,43,66,76,110,104,77,79,104,89,113,101,110,86,119,102,102,53,120,87,48,65,49,78,112,88,68,78,70,53,107,73,73,122,74,120,102,113,56,55,108,107,83,76,111,101,56,66,55,87,102,119,61,61,34,44,34,116,97,103,34,58,34,74,66,102,65,116,79,97,107,75,102,88,100,72,99,104,122,88,78,89,84,117,65,61,61,34,125]
        message_V2_pack_bytes = bytearray(demoList)

        print(" ")
        print("----------------- START Message Unpack ------------------")
        print(" ")
        messageUnpack,recipientUnpack, senderUnpack = await test_moduleV1.unpack_message(session, message_V1Service_pack_anon)
        print(" ")
        print("Unpacked message : " + messageUnpack.decode())
        print(" ")
        print("Recipient verkey : "+ recipientUnpack)
        print(" ")
        if(senderUnpack is not None):
            print("Sender Verkey : " + senderUnpack)
        else:
            print("Sender Verkey : null")
        print(" ")
        print("-------------------- FINISH Message Unpack ---------------------")
        print(" ")


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
