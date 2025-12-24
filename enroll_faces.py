from core.folder_watcher import FolderWatcher


def main():
    print("=" * 50)
    print("FACE ENROLLMENT SERVICE STARTED")
    print("=" * 50)

    watcher = FolderWatcher(image_dir="Face_images")
    watcher.run()

    print("=" * 50)
    print("FACE ENROLLMENT SERVICE COMPLETED")
    print("=" * 50)


if __name__ == "__main__":
    main()
