import numpy as np
import matplotlib.pyplot as plt
from guppi import guppi
from numba import njit
import rich
import os
import contextlib
import argparse
from scipy import signal
from sigmf import SigMFFile
from sigmf.utils import get_data_type_str
import sigmf


def main():
    fname = '/mnt/buf0/sobernberger/MAVEN/guppi_59688_63255_3379455_mars_0001-beam0000.0000.raw'#/mnt/buf0/sobernberger/1960MHz/guppi_59677_73119_16564636_AzEl_0001-beam0000.0000.raw'
    f = guppi.Guppi(fname)
    hdr = f._parse_header()
    #rich.print(hdr)
    parser = argparse.ArgumentParser(description="Convert the ATA's beamformer output raw file to IQ")
    parser.add_argument('-fc', '--f_c', type=float, help='Frequency which will be shifted to DC in MHz. Default is '\
                        +str(hdr['OBSBW']/2)+ 'MHz, as this is the center of the recorded band. Has to be a multiple of '+str(hdr['CHAN_BW'])+'.',\
                            nargs='?', default=hdr['OBSBW']/2)

    parser.add_argument('-decimation', '--decim', type=int, help='Decimation factor. Decreases outbut bandwidth to reduce file size.'\
                        ' Has to be an integer larger or equal to 2.', nargs='?', default=1)

    parser.add_argument('-X', '--X-Pol', help='Choose X polarisation to process. This is the default option.', action='store_true', default=True)
    parser.add_argument('-Y', '--Y-Pol', help='Choose Y polarisation to process.', action='store_true', default=False)

    requiredNamed = parser.add_argument_group('required named arguments')
    requiredNamed.add_argument('-i', '--Input_File_Path', type=str, help='Enter Path to input raw file', required=True)
    requiredNamed.add_argument('-o', '--Output_File_Path', type=str, help='Enter Path to interleaved IQ data and the name you wish the file to have (without file'+\
                        ' format extention)', required=True)
    
    args = parser.parse_args()


    if args.decim < 1:
        raise Exception('Decimation Factor "'+str(args.decim)+'", is invalid')

    if args.Y_Pol:
        pol = 1
        pol_str = 'Y'
    else:
        pol = 0
        pol_str = 'X'

    f_c = np.float64(float(args.f_c)*1e6)
    print(args.f_c % hdr['CHAN_BW'])
    if args.f_c % hdr['CHAN_BW'] != 0:
        fast_flag = 0
        raise Exception('Shifting Frequency must me a multiple of ' + str(hdr['CHAN_BW'])+'MHz')
    else:
        fast_flag = 1
    g = guppi.Guppi(fname)
    c = hdr['NBITS']*4*hdr['PIPERBLK']*hdr['NCHAN']
    file_size = (os.path.getsize(fname))*8
    max_blocks = round(file_size/(c+hdr['HEADER_SIZE']*8))
    num_blocks = int(input('How many Blocks do you want to process? \nEach Block consists of '+\
                            str(hdr['TBIN']*hdr['PIPERBLK']*1e3)+'ms worth of data.\nThe amount'+\
                            ' of available Blocks is '+str(max_blocks)+': '))

    if num_blocks > max_blocks or num_blocks<=0:
        raise Exception('Amount of Blocks: "'+str(num_blocks)+'", is invalid')
           
    roll_factor = -int(f_c/hdr['CHAN_BW'])
    #file_path = '/mnt/buf0/sobernberger/MAVEN/'
    print(args.Input_File_Path)
    print(args.Output_File_Path)
    write_file = open(args.Input_File_Path+'.sigmf-data', 'bw')
    for block_idx in range(num_blocks):
        ts_IQ = np.zeros(hdr['PIPERBLK']*hdr['NCHAN'], dtype='complex')
        print('Current Block: '+str(block_idx+1))
        with contextlib.redirect_stdout(None):
            hdr, data = g.read_next_block()
        
        rf_sample_idx = 0
        for spectrum_idx in range(data.shape[1]):
            rf_spectrum = data[:, spectrum_idx, pol]
            iq_spectrum = np.roll(rf_spectrum, roll_factor)
            ts_buf = np.fft.ifft(iq_spectrum)
            
            for sample in ts_buf:
                ts_IQ[rf_sample_idx] = sample
                rf_sample_idx += 1

        if args.decim > 1:
            ts_IQ = signal.decimate(ts_IQ, args.decim, ftype='fir')

        ts_IQ_interleaved = interleave(ts_IQ)
        write_file.write(ts_IQ_interleaved.tobytes())

    write_file.close()


    bandwidth = hdr['OBSBW']/args.decim

    center_frequency = hdr['OBSFREQ'] + hdr['CHAN_BW']/2 - (hdr['OBSBW']/2-args.f_c)

    meta = SigMFFile(
    data_file=args.Output_File_Path+'.sigmf-data',
    global_info = {
        SigMFFile.DATATYPE_KEY: get_data_type_str(ts_IQ_interleaved),
        SigMFFile.SAMPLE_RATE_KEY: (round(bandwidth, 2)),
        SigMFFile.AUTHOR_KEY: 'Sebastian Obernberger',
        SigMFFile.DESCRIPTION_KEY: 'Interleaved ATA Beamformer IQ capture.',
        SigMFFile.VERSION_KEY: sigmf.__version__,
        SigMFFile.FREQUENCY_KEY: center_frequency,
        'Number of Blocks': str(num_blocks),
        'Observation Time': str(hdr['TBIN']*hdr['PIPERBLK']*num_blocks*1e3)+'ms',
        'Polarisation': pol_str,
        'Original RAW header': hdr,
    }
    
    )
    assert meta.validate()
    meta.tofile(args.Output_File_Path+'.sigmf-meta')



    print('FEDISCH!! Final Bandwidth is: '+str(round(bandwidth, 2))+'MHz. Data has been shifted by: '\
            +str(round(args.f_c, 2))+'. Polarisation: '+pol_str+'.')



def interleave(ts_IQ):
    output = np.append(ts_IQ.real, ts_IQ.imag)
    output[::2] = ts_IQ.real
    output[1::2] = ts_IQ.imag
    output = np.array(output, dtype='float32')
    return output

if __name__ == "__main__":
    
    main()