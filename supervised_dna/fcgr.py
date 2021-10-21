"""
This script assumes that each Fasta file contains only one sequence
"""
from tqdm import tqdm
import random ; random.seed(42)
from pathlib import Path
from Bio import SeqIO
from complexcgr import FCGR

from .monitor_values import (
    MonitorValues, 
)

#TODO: tqdm for FCGR generation
class GenerateFCGR: 

    def __init__(self, destination_folder: Path = "img", kmer: int = 8): 
        self.destination_folder = Path(destination_folder)
        self.kmer = kmer
        self.fcgr = FCGR(kmer)
        self.counter = 0 # count number of time a sequence is converted to fcgr
        
        # Monitor Values
        self.mv = MonitorValues(["id_seq","path","path_save","len_seq",
                                "count_A","count_C","count_G","count_T"])

        # Create destination folder if needed
        self.destination_folder.mkdir(parents=True, exist_ok=True)

    def __call__(self, list_fasta):
        
        for path in tqdm(list_fasta, desc="Generating FCGR"):
            self.from_fasta(path)

        # save metadata
        self.mv.to_csv(self.destination_folder.joinpath("fcgr-metadata.csv"))

    def from_fasta(self, path: Path,):
        """FCGR for a sequence in a fasta file.
        The FCGR image will be save in 'destination_folder/specie/label/id_fasta.jpg'
        """
        # load fasta file
        path = Path(path)
        fasta  = self.load_fasta(path)
        record = next(fasta)

        # get basic information
        seq     = record.seq
        id_seq  = record.id.replace("/","_")
        len_seq = len(seq)
        count_A = seq.count("A")
        count_C = seq.count("C")
        count_G = seq.count("G")
        count_T = seq.count("T")
        
        # Generate and save FCGR for the current sequence
        _, specie, _, label  = str(path.parents[0]).split("/")
        id_fasta = path.stem
        path_save = self.destination_folder.joinpath("{}/{}/{}.jpg".format(specie, label, id_fasta))
        path_save.parents[0].mkdir(parents=True, exist_ok=True)
        self.from_seq(record.seq, path_save)
        
        # Collect values to monitor 
        self.mv()
        
    def from_seq(self, seq: str, path_save):
        "Get FCGR from a sequence"
        if not Path(path_save).is_file():
            seq = self.preprocessing(seq)
            chaos = self.fcgr(seq)
            self.fcgr.save(chaos, path_save)
        self.counter +=1

    def reset_counter(self,):
        self.counter=0
        
    @staticmethod
    def preprocessing(seq):
        seq = seq.upper()
        for letter in "BDEFHIJKLMOPQRSUVWXYZ":
            seq = seq.replace(letter,"N")
        return seq

    @staticmethod
    def load_fasta(path: Path):
        return SeqIO.parse(path, "fasta")