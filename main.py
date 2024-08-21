import streamlit as st
import pandas as pd
import numpy as np

def main():
    st.title("Phaseshift Prediction")
    
    # load the lookup table lookup_table.csv
    lookup_table = pd.read_csv('lookup_table.csv')
    
    # find unique values for the idx column
    unique_idx = lookup_table['idx'].unique()
    
    # when the user selects an idx, show the corresponding frequency, duty cycle, bmax, and phase shift
    idx = st.selectbox("Select the test index from the magnet challenge dataset", unique_idx)
    
    #print the freq dc, bmax pshift of the selected idx
    freq = lookup_table[lookup_table['idx'] == idx]['freq'].unique()
    dc = lookup_table[lookup_table['idx'] == idx]['dc'].unique()
    bmax = lookup_table[lookup_table['idx'] == idx]['bmax'].unique()
    pshift = lookup_table[lookup_table['idx'] == idx]['pshift'].unique()
    
    #sort the values
    freq.sort()
    dc.sort()
    bmax.sort()
    pshift.sort()
    
    st.write("Frequency:", freq[0], "Duty Cycle:", dc[0], "Bmax:", bmax[0], "Phase Shift:", pshift[0])
    
    pshift = st.select_slider("Phase Shift", options=pshift, value=pshift[3])
    

if __name__ == "__main__":
    main()
