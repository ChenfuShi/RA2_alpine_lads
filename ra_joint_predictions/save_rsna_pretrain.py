import os

from model.rsna_multioutput_model import save_rsna_NASnet_multioutupt

if __name__ == '__main__':
    os.chdir('/mnt/jw01-aruk-home01/projects/ra_challenge/RA_challenge/RA2_alpine_lads/')

    save_rsna_NASnet_multioutupt()
