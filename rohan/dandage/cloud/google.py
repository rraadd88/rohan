"""
pip install --upgrade google-api-python-client google-auth-httplib2 google-auth-oauthlib
"""

def get_service_drive():
    from getpass import getpass
    client_config=eval(getpass())
    from googleapiclient.discovery import build
    from google_auth_oauthlib.flow import InstalledAppFlow
    SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
    flow = InstalledAppFlow.from_client_config(client_config,SCOPES)
    creds = flow.run_console(port=0)
    service = build('drive', 'v3', credentials=creds)
    return service

def search_filetype_in_folder(folderid,filetype,service,test=False):
    filetype2mimetype={"audio":"application/vnd.google-apps.audio", #
    "document":"application/vnd.google-apps.document", #Google Docs
    "drive":"application/vnd.google-apps.drive-sdk", #3rd party shortcut
    "drawing":"application/vnd.google-apps.drawing", #Google Drawing
    "file":"application/vnd.google-apps.file", #Google Drive file
    "folder":"application/vnd.google-apps.folder", #Google Drive folder
    "form":"application/vnd.google-apps.form", #Google Forms
    "fusiontable":"application/vnd.google-apps.fusiontable", #Google Fusion Tables
    "map":"application/vnd.google-apps.map", #Google My Maps
    "photo":"application/vnd.google-apps.photo", #
    "presentation":"application/vnd.google-apps.presentation", #Google Slides
    "script":"application/vnd.google-apps.script", #Google Apps Scripts
    "shortcut":"application/vnd.google-apps.shortcut", #Shortcut
    "site":"application/vnd.google-apps.site", #Google Sites
    "spreadsheet":"application/vnd.google-apps.spreadsheet", #Google Sheets
    "unknown":"application/vnd.google-apps.unknown", #
    "video":"application/vnd.google-apps.video",}

    results = service.files().list(q=f"'{folderid}' in parents and mimeType='{filetype2mimetype[filetype]}'",
        fields="nextPageToken, files(id, name)",).execute()
    items = results.get('files', [])
    name2id={d['name']:d['id'] for d in items}
    if not items:
        print('No files found.')
    else:
        if test:
            print(name2id)
    return name2id
"""
pip install --upgrade google-api-python-client google-auth-httplib2 google-auth-oauthlib
"""

def download_file(service,fileid,filetypes,outp,test=False):
    """
    https://developers.google.com/drive/api/v3/ref-export-formats
    """
    from googleapiclient.http import MediaIoBaseDownload
    from os import makedirs
    import io
    # download
    for mimeType in filetypes:
        request = service.files().export_media(fileId=fileid,
                                                     mimeType=mimeType)
        outp=outp if '.' in outp else f"{outp}.{mimeType.split('/')[1].split('+')[0]}"
        makedirs(dirname(outp),exist_ok=True)
        fh = io.FileIO(file=outp,mode='w')
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while done is False:
            status, done = downloader.next_chunk()
            if test:
                print( f"Download {int(status.progress() * 100)}")
                
                
def download_drawings(folderid,outd,service=None,test=False):
    if service is None:
        service=get_service_drive()
    filename2id=search_filetype_in_folder(service=service,
                              filetype='drawing',
                              folderid=folderid,
                              test=test,)           
    for n in filename2id:
        download_file(service,filename2id[n],['image/png','image/svg+xml'],f"{outd}/{n}",test=test)
                