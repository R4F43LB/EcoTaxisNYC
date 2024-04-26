import pandas as pd
from openpyxl import load_workbook
from datetime import date
from datetime import datetime
from google.cloud import storage
from google.cloud import bigquery
import gcsfs
import re
import io
import numpy as np

def move_blob(bucket_name, blob_name, destination_bucket_name, destination_blob_name):
    """Moves a blob from one bucket to another with a new name."""
    # The ID of your GCS bucket
    # bucket_name = "your-bucket-name"
    # The ID of your GCS object
    # blob_name = "your-object-name"
    # The ID of the bucket to move the object to
    # destination_bucket_name = "destination-bucket-name"
    # The ID of your new GCS object (optional)
    # destination_blob_name = "destination-object-name"

    storage_client = storage.Client()

    source_bucket = storage_client.bucket(bucket_name)
    source_blob = source_bucket.blob(blob_name)
    destination_bucket = storage_client.bucket(destination_bucket_name)

    blob_copy = source_bucket.copy_blob(source_blob, destination_bucket, destination_blob_name)
    source_bucket.delete_blob(blob_name)

    print(
        "Blob {} in bucket {} moved to blob {} in bucket {}.".format(
            source_blob.name,
            source_bucket.name,
            blob_copy.name,
            destination_bucket.name
        )
    )

def gcs_trigger_xlsx_loader(event, context):
    print('Entrando a la funcion gcs_trigger_xlsx_loader')
    filename = event
    
    #the filename has a pattern of sampledatahockey.*xlsx. Process the file only when the pattern macth
    pattern = re.compile(r'sampledatahockey.*xlsx$')
    if pattern.match(event['name']): #event['name'] contains the name part of the event, here the filename
        print('Reading file {file}'.format(file=event['name']))
        
        try:
            print('Process execution started')
            hockey_file_load(filename)
            print('Process execution completed')
        
        except:
            print('Process failed')
            exit(1)
    else :
        print(f"Error encountered. Filename does not match with 'sampledatahockey.xlsx'")


def hockey_file_load(filename):
    SRC_BUCKET_NAME_PARAM = 'source-file-landing-bucket7' #the bucket where the file is supposed to land
    OP_BUCKET_NAME_PARAM = 'my-org-processing-bucket7' #the bucket where the file is supposed to process and archive
    PROJECT_NAME_PARAM = 'sturdy-gate-417001' #the GCP project name
    fs = gcsfs.GCSFileSystem(project=PROJECT_NAME_PARAM)
    client = storage.Client()
    bucket = client.get_bucket(OP_BUCKET_NAME_PARAM)
    
    #declare the filepaths
    absolute_path = 'gs://{bucket_name}/'.format(bucket_name=OP_BUCKET_NAME_PARAM)
    input_file_path = 'processing/' + filename['name']
    output_file_path = absolute_path + 'output/sampledatahockey_processed.csv'
    
    #the tabs which we are supposed to read. 
    required_tabs = ['PlayerData']
    today = datetime.today().strftime("%Y-%m-%d %H:%M:%S")
    
    #copy the file from the source bucket to the processing bucket
    move_blob(
        bucket_name=SRC_BUCKET_NAME_PARAM,
        blob_name=filename['name'],
        destination_bucket_name=OP_BUCKET_NAME_PARAM,
        destination_blob_name='processing/'+filename['name']
    )
    
    #GCS cannot read a direct GCS filepath, so it must be declared as an instance of io.BytesIO
    #class io.BytesIO() is a class in Python's io module that creates an in-memory binary stream
    #that can be used to read from or write to bytes-like objects. It provides an interface similar
    #to that of a file object, allowing you to manipulate the contents of the stream using methods
    #such as write(), read(), seek(), and tell()
    #One common use case for io.BytesIO() is when you want to work with binary data in memory
    #instead of having to create a physical file on disk. 
    
    blob = bucket.blob(input_file_path)
    buffer = io.BytesIO()
    blob.download_to_file(buffer)
    
    #reading each tab in the list
    for sheet in required_tabs:
        wb = load_workbook(buffer)
        ws = wb[sheet]
        
        #Simple pandas operation
        print('Started processing sheet {}'.format(sheet))
        df = pd.read_excel(buffer, sheet, header=[2])
        df.columns = [x.lower() for x in df.columns]
        df.insert(0, "extract_date", today, True)
        df.drop_duplicates(inplace = True)
        print('Completed processing sheet {}'.format(sheet))
    
    
    #This part highlights the process if you want the file as a CSV output 
    #logic starts here
    client = storage.Client()
    
    extract_datetime = str(datetime.today().strftime("%Y-%m-%d %H%M%S"))
    source_dir = 'processing/'
    output_dir = 'output/'
    target_dir = 'archiving/archived_files_' + extract_datetime + '/'
    
    file_name_src = list(client.list_blobs(OP_BUCKET_NAME_PARAM, prefix=source_dir))
    file_name_opt = list(client.list_blobs(OP_BUCKET_NAME_PARAM, prefix=output_dir))
    
    print('Archiving files')
    for file_name in client.list_blobs(OP_BUCKET_NAME_PARAM, prefix=source_dir):
        filename = str(file_name.name)
        filename = filename.split('/')[-1]
        print(filename)
        
        pattern_src = re.compile(r'sampledatahockey.*xlsx$')
        if pattern_src.match(filename):
            move_blob(
                bucket_name=OP_BUCKET_NAME_PARAM,
                blob_name='processing/'+filename,
                destination_bucket_name=OP_BUCKET_NAME_PARAM,
                destination_blob_name=target_dir+filename
            )
        else:
            pass
            
    for file_name in client.list_blobs(OP_BUCKET_NAME_PARAM, prefix=output_dir):
        filename = str(file_name.name)
        filename = filename.split('/')[-1]
        print(filename)
        
        pattern_src = re.compile(r'sampledatahockey_processed.*csv$')
        if pattern_src.match(filename):
            move_blob(
                bucket_name=OP_BUCKET_NAME_PARAM,
                blob_name='output/'+filename,
                destination_bucket_name=OP_BUCKET_NAME_PARAM,
                destination_blob_name=target_dir+filename
            )
        else:
            pass    
        
    df.to_csv(output_file_path, index=False)
    print('Generated files for Hockey data :{}'.format(output_file_path))
    #logic ends here
    
    
    #Direct load to BigQuery from the dataframe
    #logic starts here
    client = bigquery.Client()
    table_id_hockey = 'sturdy-gate-417001.data_clean.hockey_data'
    job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
    
    job = client.load_table_from_dataframe(df, table_id_hockey, job_config=job_config)
    job.result()
    
    table = client.get_table(table_id_hockey)
    print('Truncated and loaded {} rows into {} table.'.format(table.num_rows, table.table_id))
    #logic ends here