import instaloader
import os

def cleanup_non_images(folder):
    image_extensions = {".jpg", ".jpeg", ".png"}
    
    for file_name in os.listdir(folder):
        file_path = os.path.join(folder, file_name)
        
        if os.path.isfile(file_path):
            _, extension = os.path.splitext(file_name)
            if extension.lower() not in image_extensions:
                try:
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")
                except Exception as e:
                    print(f"Failed to delete {file_path}: {e}")

def download_instagram_photos(username, max_images=10):
    loader = instaloader.Instaloader()

    try:
        print(f"Downloading up to {max_images} photos from @{username}...")
        profile = instaloader.Profile.from_username(loader.context, username)
        images_downloaded = 0
        
        # Iterate through posts in the profile
        for post in profile.get_posts():
            if images_downloaded >= max_images:
                break
            
            loader.download_post(post, target=username)
            images_downloaded += 1
            print(f"Downloaded {images_downloaded}/{max_images} images.")
        
        print(f"Photos downloaded successfully into the '{username}' folder.")
        cleanup_non_images(username)
    except instaloader.exceptions.ProfileNotExistsException:
        print(f"Error: The profile @{username} does not exist.")
    except instaloader.exceptions.PrivateProfileNotAccessibleException:
        print(f"Error: The profile @{username} is private. Cannot access data.")
    except Exception as e:
        print(f"An error occurred: {e}")

download_instagram_photos("ucu_apps", 10)
