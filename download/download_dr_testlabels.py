import requests

url = "https://storage.googleapis.com/kaggle-forum-message-attachments/90528/2877/retinopathy_solution.csv"
    
save_path = "/home/pgshare/shared/datasets/dr250/testLables.csv"

response = requests.get(url)
if response.status_code == 200:
    with open(save_path, 'wb') as file:
        file.write(response.content)
    print(f"File successfully downloaded and saved to {save_path}")
else:
    print(f"Failed to download file. Status code: {response.status_code}")
