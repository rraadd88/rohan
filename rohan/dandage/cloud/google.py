"""
pip install --upgrade google-api-python-client google-auth-httplib2 google-auth-oauthlib
"""

def get_service(service_name='drive',access_limit=True,client_config=None):
    service_name2params={
    'drive': {
        'scope':'https://www.googleapis.com/auth/drive.readonly',
        'build':'drive',
        'version': 'v3'},        
    'slides': {
        'scope':'https://www.googleapis.com/auth/presentations',
        'build':'slides',
        'version': 'v1'},
    'sheets':{
        'scope':'https://www.googleapis.com/auth/spreadsheets',
        'build':'sheets',
        'version': 'v4'}}
    if client_config is None:
        from getpass import getpass
        client_config=eval(getpass())
    from googleapiclient.discovery import build
    from google_auth_oauthlib.flow import InstalledAppFlow
    SCOPES = [service_name2params[service_name]['scope']]
    if not access_limit:
        SCOPES = [s.replace('.readonly','') for s in SCOPES]
    flow = InstalledAppFlow.from_client_config(client_config,SCOPES)
    creds = flow.run_console(port=0)
    service = build(service_name,service_name2params[service_name]['version'], credentials=creds)
    return service

get_service_drive=get_service
    
def list_files_in_folder(service,folderid,filetype=None,test=False):
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

    results = service.files().list(
        q=f"'{folderid}' in parents"+(f" and mimeType='{filetype2mimetype[filetype]}'" if not filetype is None else ""),
        fields="nextPageToken, files(id, name)",).execute()
    items = results.get('files', [])
    name2id={d['name']:d['id'] for d in items}
    if not items:
        print('No files found.')
    else:
        if test:
            print(name2id)
    return name2id

def download_file(service,fileid,filetypes,outp,test=False):
    """
    https://developers.google.com/drive/api/v3/ref-export-formats
    """
    from googleapiclient.http import MediaIoBaseDownload
    from os import makedirs
    from os.path import dirname
    import io
    # download
    for mimeType in filetypes:
        request = service.files().export_media(fileId=fileid,
                                                     mimeType=mimeType)
        outp_=outp if '.' in outp else f"{outp}.{mimeType.split('/')[1].split('+')[0]}"
        makedirs(dirname(outp),exist_ok=True)
        fh = io.FileIO(file=outp_,mode='w')
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while done is False:
            status, done = downloader.next_chunk()
            if test:
                print( f"Downloading {outp_}: {int(status.progress() * 100)}")
                
def upload_file(service,filep,folder_id,test=False):
    from googleapiclient.http import MediaFileUpload
    from os.path import basename
    file_metadata = {'name': basename(filep),
                    'parents': [folder_id]}
    media = MediaFileUpload(filep, mimetype=f"image/{filep.split('.')[1]}")
    file_name2id=list_files_in_folder(service,folderid=folder_id,filetype=None,test=False)
    if not basename(filep) in file_name2id:
        file = service.files().create(body=file_metadata,
                                        media_body=media,
                                        fields='id').execute()
        file_id=file['id']
    else:
        file_id=file_name2id[basename(filep)]
        if test:
            print(f"replacing {basename(filep)} in the folder")
        file=service.files().get(fileId=file_name2id[basename(filep)]).execute()
        del file['id']
        response = service.files().update(fileId=file_id,
                                          body=file,
                                          media_body=media,
                                         ).execute()
    return file_id
def download_drawings(folderid,outd,service=None,test=False):
    if service is None:
        service=get_service_drive()
    filename2id=list_files_in_folder(service=service,
                              filetype='drawing',
                              folderid=folderid,
                              test=test,)           
    for n in filename2id:
        download_file(service,filename2id[n],[
#                                                 'image/png',
                                              'image/svg+xml'],
                      f"{outd}/{n}",test=test)

class slides():
    def get_page_ids(service,presentation_id):
        presentation = service.presentations().get(
            presentationId=presentation_id).execute()
        slides = presentation.get('slides')
        print('The presentation contains {} slides:'.format(len(slides)))
        return [slide.get('objectId') for i, slide in enumerate(slides)]

    def create_image(service, presentation_id, page_id,image_id):
        """
        image less than 1.5 Mb
        """
        import numpy as np
        IMAGE_URL = (f'https://drive.google.com/uc?id={image_id}')
        image_id = f'image_{np.random.rand()}'.replace('.','')
        size = {
            'magnitude': 576,
            'unit': 'PT'
        }
        requests = []
        requests.append({
            'createImage': {
                'objectId': image_id,
                'url': IMAGE_URL,
                'elementProperties': {
                    'pageObjectId': page_id,
                    'size': {
                        'height': size,
                        'width': size
                    },
                    'transform': {
                        'scaleX': 1,
                        'scaleY': 1,
                        'unit': 'EMU'
                    }
                }
            }
        })
        # Execute the request.
        body = {
            'requests': requests
        }
        response = service.presentations() \
            .batchUpdate(presentationId=presentation_id, body=body).execute()
        create_image_response = response.get('replies')[0].get('createImage')
#         print('Created image with ID: {0}'.format(
#             create_image_response.get('objectId')))

        # [END slides_create_image]
        return create_image_response.get('objectId')
#     def update_images(presentation_id,page_id2image_id):
#         create_image(service, presentation_id, page_id,image_id)
#         page_ids=get_page_ids(service,presentation_id)
#         zip(page_ids)


def get_comments(fileid,fields='comments/quotedFileContent/value,comments/content,comments/id'):
    """
    fields: comments/
                kind:
                id:
                createdTime:
                modifiedTime:
                author:
                    kind:
                    displayName:
                    photoLink:
                    me:
                        True
                htmlContent:
                content:
                deleted:
                quotedFileContent:
                    mimeType:
                    value:
                anchor:
                replies:
                    []
    """
    def apply_(service,**kws_list)
        comments = service.comments().list(**kws_list).execute()
        df1=pd.DataFrame(pd.concat({di:pd.Series({k:d[k] for k in d}) for di,d in enumerate(comments['comments'])},
                 axis=0)).reset_index().rename(columns={'level_0':'comment #',
                                                        'level_1':'key',0:'value'})
        df1['value']=df1['value'].apply(lambda x: ','.join(x.values()) if isinstance(x,dict) else x)
        df1=df1.set_index(['comment #','key'])
        df1=df1.unstack(1).droplevel(0,1)
        df1['link']=df1['id'].apply(lambda x: f"https://drive.google.com/file/d/{fileid}/edit?disco={x}")
        df1=df1.rename(columns={'content':'comment',
                           'quotedFileContent':'text'}).drop(['id'],axis=1)
    service=get_service()    
    if not isinstance(fileid,str):
        fileid=[fileid]
    df1=pd.concat([apply_(service,fileId=fileid,
                    #fields='comments',
                     fields=fields,# nextPageToken',
                     includeDeleted='false',
                     pageSize=100) for k in fileid],
             axis=0)
    return df1