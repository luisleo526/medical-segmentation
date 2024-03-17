import os

from boxsdk import OAuth2, Client


def download_file(file, dest_folder):
    file_name = file.name
    file_path = os.path.join(dest_folder, file_name)

    with open(file_path, 'wb') as f:
        file.download_to(f)


def download_folder(folder, dest_folder):
    folder_name = folder.name
    folder_path = os.path.join(dest_folder, folder_name)

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    for item in folder.get_items():
        if item.type == 'file':
            download_file(item, folder_path)
        elif item.type == 'folder':
            download_folder(item, folder_path)


if __name__ == '__main__':
    client_id = 'v4pc4be3bo81jcwf4f45ayc6d0t3eits'
    client_secret = 'xAlcWVOClfd7ut6ZguaybzLs5wu9OoUW'
    access_token = 'tpEoYcbgW2EOJDcR7oN690xQgkacy2ti'

    auth = OAuth2(
        client_id=client_id,
        client_secret=client_secret,
        access_token=access_token
    )
    client = Client(auth)

    folder_id = '254052087637'
    dest_folder = '.'

    folder = client.folder(folder_id).get()
    download_folder(folder, dest_folder)
