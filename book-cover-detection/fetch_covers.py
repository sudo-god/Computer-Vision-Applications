import random
import requests
from PIL import Image
from io import BytesIO


# book_ids = ["1851094903", "0415972353", "0618714898", "0787665703", "1573563021", "9781612680194", "9788190565585", "9780091929114"] # test book isbns 
book_ids = ["0618714898"] # test book isbns 
# book_ids = range(1, 93)
for book_id in book_ids:
    response = requests.get(f'https://covers.openlibrary.org/b/isbn/{book_id}-L.jpg')
    image = Image.open(BytesIO(response.content))
    image.save(f'./cover_{book_id}.jpg')
