import os
import csv
import re
from typing import List


class CSVFileLoader:
    def __init__(self, path: str, encoding: str = "utf-8"):
        self.documents = []
        self.path = path
        self.encoding = encoding

    def load(self):
        if not os.path.isfile(self.path) or not self.path.endswith(".csv"):
            raise ValueError("Provided path is not a valid .csv file.")
        
        self.load_file()

    def parse_wkt(self, wkt: str):
        # Example WKT: "POINT (-84.2484845 34.5581409)"
        match = re.match(r"POINT \((-?\d+\.\d+) (-?\d+\.\d+)\)", wkt)
        if match:
            lon, lat = match.groups()
            return float(lat), float(lon)
        return None, None

    def load_file(self):
        with open(self.path, "r", encoding=self.encoding) as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = row.get("name", "").strip()
                description = row.get("description", "").strip()
                wkt = row.get("WKT", "").strip()
                lat, lon = self.parse_wkt(wkt)
                if name or description:
                    document = {
                        "text": f"Location: {name}\nDescription: {description}",
                        "name": name,
                        "description": description,
                        "wkt": wkt,
                        "latitude": lat,
                        "longitude": lon
                    }
                    self.documents.append(document)

    def load_documents(self):
        self.load()
        return self.documents


class CharacterTextSplitter:
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        assert (
            chunk_size > chunk_overlap
        ), "Chunk size must be greater than chunk overlap"

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, text: str) -> List[str]:
        chunks = []
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            chunks.append(text[i : i + self.chunk_size])
        return chunks

    def split_texts(self, texts: List[str]) -> List[str]:
        chunks = []
        for text in texts:
            chunks.extend(self.split(text))
        return chunks
