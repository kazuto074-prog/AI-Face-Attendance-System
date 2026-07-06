import face_recognition
import os
import pickle

# ─────────────────────────────────────────
#  SETTINGS  — change these if needed
# ─────────────────────────────────────────
KNOWN_FACES_DIR = "known_faces"   # folder with subfolders per person
ENCODINGS_FILE  = "encodings.pkl" # output file — your face database

# ─────────────────────────────────────────
#  STORAGE
# ─────────────────────────────────────────
known_encodings = []
known_names     = []

print("=" * 45)
print("   Face Encoding Script")
print("   Scanning all folders in known_faces")
print("=" * 45)
print()

# Check the known_faces folder exists
if not os.path.exists(KNOWN_FACES_DIR):
    print(f"ERROR: '{KNOWN_FACES_DIR}' folder not found!")
    print("Make sure you are running this script from inside")
    print("your face_recognition_project folder.")
    exit()

# ─────────────────────────────────────────
#  LOOP THROUGH EACH PERSON'S FOLDER
# ─────────────────────────────────────────
for person_name in os.listdir(KNOWN_FACES_DIR):
    person_folder = os.path.join(KNOWN_FACES_DIR, person_name)

    # Skip if it's not a folder
    if not os.path.isdir(person_folder):
        continue

    print(f"Processing: {person_name.upper()}")
    print(f"Folder: {person_folder}")

    photo_count = 0

    # Loop through every image in the person's folder
    for filename in os.listdir(person_folder):

        # Only process image files
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        img_path = os.path.join(person_folder, filename)
        print(f"  Encoding: {filename} ... ", end="")

        # Load the image
        image = face_recognition.load_image_file(img_path)

        # Find face encodings in the image
        encodings = face_recognition.face_encodings(image)

        # Handle: no face found
        if len(encodings) == 0:
            print("WARNING — No face found, skipping.")
            continue

        # Handle: multiple faces found (use first one only)
        if len(encodings) > 1:
            print("WARNING — Multiple faces found, using first face only.")
        else:
            print("OK")

        # Save the encoding and the person's name
        known_encodings.append(encodings[0])
        known_names.append(person_name)
        photo_count += 1

    print(f"  → {photo_count} photo(s) encoded for {person_name}")
    print()

# ─────────────────────────────────────────
#  SAVE TO DISK
# ─────────────────────────────────────────
if len(known_encodings) == 0:
    print("ERROR: No faces were encoded!")
    print("Check that your photos are inside the correct folders:")
    print("  known_faces/suhaib/  — add Suhaib's photos here")
    print("  known_faces/ahmed/   — add Ahmed's photos here")
    exit()

    print("\nEncoding Summary")
    print("----------------")
    for person in sorted(set(known_names)):
        count = known_names.count(person)
    print(f"{person}: {count} encoding(s)")
    with open(ENCODINGS_FILE, "wb") as f:
        pickle.dump({
        "encodings": known_encodings,
        "names":     known_names
    }, f)

# ─────────────────────────────────────────
#  SUMMARY
# ─────────────────────────────────────────
print("=" * 45)
print(f"  Done! {len(known_names)} face encoding(s) saved.")
print(f"  People in database: {list(set(known_names))}")
print(f"  Saved to: {ENCODINGS_FILE}")
print("  Ready for Phase 4!")
print("=" * 45)
