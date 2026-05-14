"""Load and return all models used by the pipeline."""

from config import DEVICE


def _freeze(model):
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def load_models() -> dict:
    """
    Downloads (first run) and loads all models into DEVICE.
    Returns a flat dict of named components.
    """
    from transformers import (
        MimiModel, AutoFeatureExtractor,
        MarianMTModel, MarianTokenizer,
        VitsModel, AutoTokenizer,
    )
    import whisper

    m = {}

    print("[load] Whisper medium ...")
    m["whisper"] = whisper.load_model("medium").to(DEVICE)

    print("[load] Mimi encoder/decoder ...")
    m["mimi_fe"] = AutoFeatureExtractor.from_pretrained("kyutai/mimi")
    m["mimi"]    = _freeze(MimiModel.from_pretrained("kyutai/mimi").to(DEVICE).eval())

    print("[load] Helsinki-NLP FR->EN ...")
    m["mt_fr_en_tok"]   = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-fr-en")
    m["mt_fr_en_model"] = _freeze(
        MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-fr-en").to(DEVICE).eval())

    print("[load] Helsinki-NLP EN->FR ...")
    m["mt_en_fr_tok"]   = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-fr")
    m["mt_en_fr_model"] = _freeze(
        MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-fr").to(DEVICE).eval())

    print("[load] MMS-TTS EN ...")
    m["tts_en_tok"]   = AutoTokenizer.from_pretrained("facebook/mms-tts-eng")
    m["tts_en_model"] = _freeze(
        VitsModel.from_pretrained("facebook/mms-tts-eng").to(DEVICE).eval())

    print("[load] MMS-TTS FR ...")
    m["tts_fr_tok"]   = AutoTokenizer.from_pretrained("facebook/mms-tts-fra")
    m["tts_fr_model"] = _freeze(
        VitsModel.from_pretrained("facebook/mms-tts-fra").to(DEVICE).eval())

    print(f"[load] All models ready  (device={DEVICE})\n")
    return m
