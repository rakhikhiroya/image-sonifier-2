from flask import Flask, request, send_file, render_template, jsonify
import os, random, colorsys, uuid
import numpy as np
from PIL import Image
import pretty_midi
import soundfile as sf
from pedalboard import Pedalboard, Reverb, Chorus
from pedalboard.io import AudioFile
from music21 import (stream, note, chord, metadata, key,
                     tempo, dynamics, instrument, expressions, meter)

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

INSTRUMENT_RANGES = {
    "Violin":      (55, 103),
    "Viola":       (48, 88),
    "Violoncello": (36, 76),
    "Flute":       (60, 96),
    "Oboe":        (58, 91),
    "Clarinet":    (50, 94),
    "Bassoon":     (34, 75),
    "Piano":       (21, 108)
}

def clamp_to_range(pitch, instrument_obj):
    name = instrument_obj.instrumentName
    low, high = INSTRUMENT_RANGES.get(name, (21, 108))
    while pitch.midi < low:
        pitch.octave += 1
    while pitch.midi > high:
        pitch.octave -= 1
    return pitch

def analyse_image(image_path, num_slices):
    img = Image.open(image_path).convert("RGB")
    width, height = img.size
    pixels = np.array(img)
    NUMBER_OF_SLICES = min(num_slices, width)
    slice_width = max(1, width // NUMBER_OF_SLICES)
    rgb_data, brightness, saturation, vertical_positions = [], [], [], []
    total_r = total_g = total_b = 0
    for i in range(NUMBER_OF_SLICES):
        x_start = i * slice_width
        x_end = min(x_start + slice_width, width)
        section = pixels[:, x_start:x_end, :]
        avg_color = section.mean(axis=(0, 1))
        r, g, b = avg_color
        rgb_data.append((r, g, b))
        total_r += r; total_g += g; total_b += b
        brightness.append(np.mean(avg_color) / 255)
        r_n, g_n, b_n = r / 255, g / 255, b / 255
        _, sat, _ = colorsys.rgb_to_hsv(r_n, g_n, b_n)
        saturation.append(sat)
        vertical_profile = section.mean(axis=2).mean(axis=1)
        vertical_positions.append(int(np.argmax(vertical_profile)))
    return (rgb_data, brightness, saturation, vertical_positions,
            height, total_r, total_g, total_b)

def determine_cadence_from_image(total_r, total_g, total_b):
    totals = {"Red": total_r, "Green": total_g, "Blue": total_b}
    sorted_colours = sorted(totals.items(), key=lambda x: x[1], reverse=True)
    dominant_colour = sorted_colours[0][0]
    dominant_value  = sorted_colours[0][1]
    second_value    = sorted_colours[1][1]
    if abs(dominant_value - second_value) < 0.05 * dominant_value:
        return "Interrupted (V-VI)", [5, 6]
    if dominant_colour == "Red":
        return "Perfect (V-I)", [5, 1]
    elif dominant_colour == "Blue":
        return "Plagal (IV-I)", [4, 1]
    else:
        return "Imperfect (ends on V)", [1, 5]

def rgb_to_degree(r, g, b, style):
    total = r + g + b + 1e-6
    r_w, g_w, b_w = r/total, g/total, b/total
    if style == "classical":
        if r_w > 0.45:   return 1
        elif g_w > 0.45: return 3
        elif b_w > 0.45: return 5
        else:            return random.choice([2, 4, 6, 7])
    else:
        return random.choice([1, 2, 3, 5, 6])

def midi_to_wav(midi_path, wav_path, add_effects, reverb_amount, chorus_amount):
    midi_data = pretty_midi.PrettyMIDI(midi_path)
    for inst in midi_data.instruments:
        try:
            inst.program = pretty_midi.instrument_name_to_program(inst.name)
        except:
            inst.program = 0
    audio = midi_data.synthesize(fs=44100, wave=np.sin)
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val
    sf.write(wav_path, audio, 44100)
    if add_effects:
        effects_path = wav_path.replace(".wav", "_effects.wav")
        board = Pedalboard([Chorus(depth=float(chorus_amount)),
                            Reverb(room_size=float(reverb_amount))])
        with AudioFile(wav_path) as f:
            raw = f.read(f.frames)
            effected = board(raw, f.samplerate)
        with AudioFile(effects_path, "w", f.samplerate, effected.shape[0]) as f:
            f.write(effected)
        return effects_path
    return wav_path

def generate_score(image_path, config):
    KEY_SIGNATURE    = config["key_signature"]
    TEMPO_BPM        = int(config["tempo"])
    STYLE            = config["style"]
    BASE_OCTAVE      = int(config["base_octave"])
    REGISTER_SPAN    = int(config["register_span"])
    NUMBER_OF_SLICES = int(config["num_slices"])
    PIECE_NAME       = config["piece_name"]
    ADD_EFFECTS      = config.get("add_effects") == "true"
    REVERB_AMOUNT    = float(config.get("reverb_amount", 0.4))
    CHORUS_AMOUNT    = float(config.get("chorus_amount", 0.3))

    instrument_map = {
        "Flute": instrument.Flute(), "Oboe": instrument.Oboe(),
        "Violin": instrument.Violin(), "Clarinet": instrument.Clarinet(),
        "Viola": instrument.Viola(), "Violoncello": instrument.Violoncello(),
        "Bassoon": instrument.Bassoon(), "Piano": instrument.Piano(),
    }
    sel_melody  = instrument_map.get(config["melody_instrument"],  instrument.Flute())
    sel_counter = instrument_map.get(config["counter_instrument"], instrument.Oboe())
    sel_harmony = instrument_map.get(config["harmony_instrument"], instrument.Piano())
    sel_bass    = instrument_map.get(config["bass_instrument"],    instrument.Violoncello())

    (rgb_data, brightness, saturation, vertical_positions,
     height, total_r, total_g, total_b) = analyse_image(image_path, NUMBER_OF_SLICES)
    cadence_type, cadence_degrees = determine_cadence_from_image(total_r, total_g, total_b)

    score = stream.Score()
    score.insert(0, metadata.Metadata())
    score.insert(0, tempo.MetronomeMark(number=TEMPO_BPM))
    score.insert(0, key.Key(KEY_SIGNATURE))
    score.insert(0, meter.TimeSignature("4/4"))
    tonal_key     = key.Key(KEY_SIGNATURE)
    scale         = tonal_key.getScale()
    scale_pitches = [scale.pitchFromDegree(d) for d in range(1, 8)]
    score.metadata.title = PIECE_NAME

    melody  = stream.Part(); melody.insert(0,  sel_melody)
    counter = stream.Part(); counter.insert(0, sel_counter)
    harmony = stream.Part(); harmony.insert(0, sel_harmony)
    bass    = stream.Part(); bass.insert(0,    sel_bass)

    rhythm_patterns = [
        [1,0.5,0.5],[0.75,0.25,0.5],[0.5,0.5,0.5,0.5],
        [1.5,0.5],[0.25,0.25,0.5,1],[0.5,1,0.5],[0.25,0.75,0.5]
    ]
    current_pattern = random.choice(rhythm_patterns)
    pattern_index = elapsed_time = 0
    current_dynamic = None
    current_bar = 1

    for i in range(NUMBER_OF_SLICES):
        r, g, b = rgb_data[i]
        degree  = rgb_to_degree(r, g, b, STYLE)
        pitch   = scale.pitchFromDegree(degree)
        octave_shift = int((1 - vertical_positions[i] / height) * REGISTER_SPAN)
        pitch.octave = BASE_OCTAVE + octave_shift
        pitch = clamp_to_range(pitch, sel_melody)

        duration = current_pattern[pattern_index]
        pattern_index += 1
        if pattern_index >= len(current_pattern):
            current_pattern = random.choice(rhythm_patterns)
            pattern_index = 0

        melody.append(note.Rest(quarterLength=duration) if random.random() < 0.12
                      else note.Note(pitch, quarterLength=duration))

        counter_degree = (degree - 2 - 1) % 7 + 1
        counter_pitch  = scale.pitchFromDegree(counter_degree)
        counter_pitch.octave = pitch.octave
        if counter_pitch.midi >= pitch.midi:
            counter_pitch.octave -= 1
        counter.append(note.Note(counter_pitch, quarterLength=duration))

        if   brightness[i] < 0.3: dyn_mark = "p"
        elif brightness[i] < 0.5: dyn_mark = "mp"
        elif brightness[i] < 0.7: dyn_mark = "mf"
        else:                      dyn_mark = "f"
        if dyn_mark != current_dynamic:
            for part in [melody, counter, harmony, bass]:
                part.insert(part.highestTime, dynamics.Dynamic(dyn_mark))
            current_dynamic = dyn_mark

        elapsed_time += duration
        if elapsed_time >= current_bar * 4:
            sat = saturation[i]
            chord_degrees = [degree, (degree+2-1)%7+1, (degree+4-1)%7+1]
            if sat >= 0.3:
                chord_degrees.append((degree+6-1)%7+1)
            chord_pitches = [scale.pitchFromDegree(d) for d in chord_degrees]
            harmony.append(chord.Chord(chord_pitches, quarterLength=3))
            harmony.append(chord.Chord(chord_pitches, quarterLength=1))
            root_pitch = scale.pitchFromDegree(degree)
            root_pitch.octave = 2
            bass.append(note.Note(root_pitch, quarterLength=3))
            bass.append(note.Note(root_pitch, quarterLength=1))
            current_bar += 1

    parts    = [melody, counter, harmony, bass]
    max_time = max(p.highestTime for p in parts)
    remainder = max_time % 4
    padding   = (4 - remainder) if remainder != 0 else 0
    for p in parts:
        if p.highestTime < max_time:
            p.append(note.Rest(quarterLength=max_time - p.highestTime))
        if padding > 0:
            p.append(note.Rest(quarterLength=padding))

    cadence_start = max(p.highestTime for p in parts)
    melody.insert(cadence_start, expressions.TextExpression("rit."))
    melody.append(note.Note(scale_pitches[4], quarterLength=4))
    final_mel = note.Note(scale_pitches[4], quarterLength=4)
    final_mel.expressions.append(expressions.Fermata())
    melody.append(final_mel)

    counter.append(note.Note(scale_pitches[3], quarterLength=4))
    for sp, ql in [(3,1),(2,1),(1,1),(2,1)]:
        counter.append(note.Note(scale_pitches[sp], quarterLength=ql))
    list(counter.flat.notes)[-1].expressions.append(expressions.Fermata())

    current_time = cadence_start
    for i, d in enumerate(cadence_degrees):
        score.insert(current_time, tempo.MetronomeMark(number=TEMPO_BPM * (0.9**(i+1))))
        root = scale.pitchFromDegree(d)
        fc = chord.Chord([root,
                          scale.pitchFromDegree((d+2-1)%7+1),
                          scale.pitchFromDegree((d+4-1)%7+1)],
                         quarterLength=4)
        if d == 1: fc.expressions.append(expressions.Fermata())
        harmony.append(fc)
        bn = note.Note(root, quarterLength=4)
        bn.octave = 2
        if d == 1: bn.expressions.append(expressions.Fermata())
        bass.append(bn)
        current_time += 4

    parts    = [melody, counter, harmony, bass]
    max_time = max(p.highestTime for p in parts)
    for p in parts:
        remaining = max_time - p.highestTime
        if remaining > 0:
            p.append(note.Rest(quarterLength=remaining))
    for p in parts:
        score.append(p)

    return score, cadence_type, ADD_EFFECTS, REVERB_AMOUNT, CHORUS_AMOUNT

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/generate", methods=["POST"])
def generate():
    if "image" not in request.files or request.files["image"].filename == "":
        return jsonify({"error": "No image uploaded"}), 400
    file = request.files["image"]
    uid  = str(uuid.uuid4())[:8]
    image_path = os.path.join(UPLOAD_FOLDER, f"{uid}_{file.filename}")
    file.save(image_path)
    config = {
        "key_signature":      request.form.get("key_signature",      "Am"),
        "tempo":              request.form.get("tempo",              "70"),
        "style":              request.form.get("style",              "cinematic"),
        "base_octave":        request.form.get("base_octave",        "4"),
        "register_span":      request.form.get("register_span",      "2"),
        "num_slices":         request.form.get("num_slices",         "150"),
        "piece_name":         request.form.get("piece_name",         "My Piece"),
        "melody_instrument":  request.form.get("melody_instrument",  "Flute"),
        "counter_instrument": request.form.get("counter_instrument", "Oboe"),
        "harmony_instrument": request.form.get("harmony_instrument", "Piano"),
        "bass_instrument":    request.form.get("bass_instrument",    "Violoncello"),
        "add_effects":        request.form.get("add_effects",        "true"),
        "reverb_amount":      request.form.get("reverb_amount",      "0.4"),
        "chorus_amount":      request.form.get("chorus_amount",      "0.3"),
        "output_format":      request.form.get("output_format",      "midi"),
    }
    try:
        score, cadence_type, add_effects, reverb_amount, chorus_amount = generate_score(image_path, config)
        midi_filename = f"{uid}_output.mid"
        midi_path     = os.path.join(OUTPUT_FOLDER, midi_filename)
        score.write("midi", midi_path)
        response_data = {"cadence": cadence_type}
        if config["output_format"] == "wav":
            wav_path   = os.path.join(OUTPUT_FOLDER, f"{uid}_output.wav")
            final_path = midi_to_wav(midi_path, wav_path, add_effects, reverb_amount, chorus_amount)
            response_data["wav_file"]  = os.path.basename(final_path)
            response_data["midi_file"] = midi_filename
        else:
            response_data["midi_file"] = midi_filename
        os.remove(image_path)
        return jsonify(response_data)
    except Exception as e:
        if os.path.exists(image_path): os.remove(image_path)
        return jsonify({"error": str(e)}), 500

@app.route("/download/<filename>")
def download(filename):
    path = os.path.join(OUTPUT_FOLDER, filename)
    if not os.path.exists(path):
        return jsonify({"error": "File not found"}), 404
    return send_file(path, as_attachment=True, download_name=filename)

if __name__ == "__main__":
    app.run(debug=True)