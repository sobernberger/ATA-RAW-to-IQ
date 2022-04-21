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
    # Initializing Parser
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
    # Reading the first header to extract some information
    f = guppi.Guppi(fname)
    hdr = f._parse_header()

    # Check if parsed arguments meet requirements
    if args.decim < 1:
        raise Exception('Decimation Factor "'+str(args.decim)+'", is invalid')

    if args.Y_Pol:
        pol = 1
        pol_str = 'Y'
    else:
        pol = 0
        pol_str = 'X'
    if args.f_c % hdr['CHAN_BW'] != 0:
        fast_flag = 0
        raise Exception('Shifting Frequency must me a multiple of ' + str(hdr['CHAN_BW'])+'MHz')
    else:
        fast_flag = 1
    f_c = np.float64(float(args.f_c)*1e6)
    fname = args.Input_File_Path

    # Read in *.raw file
    g = guppi.Guppi(fname)

    # Calculate number of available blocks, using file size, header size and number of bits per sample
    c = hdr['NBITS']*4*hdr['PIPERBLK']*hdr['NCHAN']
    file_size = (os.path.getsize(fname))*8
    max_blocks = round(file_size/(c+hdr['HEADER_SIZE']*8))

    # Getting desired amount of blocks to process
    num_blocks = int(input('How many Blocks do you want to process? \nEach Block consists of '+\
                            str(hdr['TBIN']*hdr['PIPERBLK']*1e3)+'ms worth of data.\nThe amount'+\
                            ' of available Blocks is '+str(max_blocks)+': '))
    # Check if the desired amount of blocks is valid
    if num_blocks > max_blocks or num_blocks<=0:
        raise Exception('Amount of Blocks: "'+str(num_blocks)+'", is invalid')
           
    # Calculate np.roll roll factor which will do the rf to iq conversion
    roll_factor = -int(f_c/hdr['CHAN_BW'])
    # Opening Output file
    write_file = open(args.Output_File_Path+'.sigmf-data', 'bw')
    # Looping through the blocks
    for block_idx in range(num_blocks):
        # Initialize Buffer Array
        ts_IQ = np.zeros(hdr['PIPERBLK']*hdr['NCHAN'], dtype='complex')
        print('Current Block: '+str(block_idx+1))
        with contextlib.redirect_stdout(None):
            # Reading in next block
            hdr, data = g.read_next_block()
        
        rf_sample_idx = 0
        for spectrum_idx in range(data.shape[1]):
            # Looping through the Spectra in one block, converting it to IQ and then to time domain
            rf_spectrum = data[:, spectrum_idx, pol]
            iq_spectrum = np.roll(rf_spectrum, roll_factor)
            ts_buf = np.fft.ifft(iq_spectrum)
            
            for sample in ts_buf:
                # Writing result of IFFT to Buffer
                ts_IQ[rf_sample_idx] = sample
                rf_sample_idx += 1
        # Decimate if applicable
        if args.decim > 1:
            ts_IQ = signal.decimate(ts_IQ, args.decim, ftype='fir')

        # Interleave and write binaries to file
        ts_IQ_interleaved = interleave(ts_IQ)
        write_file.write(ts_IQ_interleaved.tobytes())
    # Close output file
    write_file.close()

    # Calculating information for sigmf meta data
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
    # Validating and writing meta data to file
    assert meta.validate()
    meta.tofile(args.Output_File_Path+'.sigmf-meta')



    print('Final Bandwidth is: '+str(round(bandwidth, 2))+'MHz. Data has been shifted by: '\
            +str(round(args.f_c, 2))+'. Polarisation: '+pol_str+'.')


# Interleaving: Complex numbers are split into real and imaginary part and then reasambled into a new array. 
# The real part of each number is followed by the imaginary part of the same number
def interleave(ts_IQ):
    output = np.append(ts_IQ.real, ts_IQ.imag)
    output[::2] = ts_IQ.real
    output[1::2] = ts_IQ.imag
    output = np.array(output, dtype='float32')
    return output

if __name__ == "__main__":
    
    main()