import cropper
import facial_thirds
import is_narrow

if __name__ == "__main__":
    print("=== STEP 1: Cropping Faces ===")
    cropper.main()

    print("\n=== STEP 2: Drawing Facial Thirds ===")
    facial_thirds.main()

    print("\n=== STEP 3: Checking Narrow Face ===")
    is_narrow.main()

    print("\n✅ All steps completed successfully!")
