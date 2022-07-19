# Corruption-Invariant-Tracking-Benchmark (CITB)

This benchmark contains VOT2020ST-C, GOT-10k-C, VOT2020-LT-C, UAV20L-C, and DepthTrack-C datasets.

## Corruptions in Tracking

## Random Corruption Robustness

![image](https://github.com/memoryunreal/Corruption-Invariant-Tracking-Benchmark/blob/main/randomuav.png)

## Corrupted Data Corruption Visualization

![image](https://github.com/memoryunreal/Corruption-Invariant-Tracking-Benchmark/blob/main/visualization.png)

## Statics of our benchmark datasets for single and multiple modality object tracking.
| Dataset       | Description         | \#Videos | Added Corruptions                               |
|---------------|---------------------|----------|-------------------------------------------------|
| VOT2020-ST    | Short-term tracking | 60       | Noise; Blur; Digital                            |
| GOT-10k       | Short-term tracking | 180      | Noise; Blur; Digital                            |
| VOT2020-LT    | Long-term tracking  | 50       | Noise; Blur; Digital; Transmission              |
| UAV20L        | UAV tracking        | 20       | Noise; Blur; Digital; Weather; Transmission     |
| DepthTrack    | RGB-Depth Tracking  | 50       | Noise; Blur; Digital; Transmission; Multi-modal |


## Tracker performance on clean data and corrupted data. Results are shown in percentage. 
<table>
    <tr>
        <th> </th><th colspan="3">VOT2020-ST</th> <th colspan="3">VOT2020-ST-C </th> <th colspan="2"> GOT-10k </th> <th colspan="2"> GOT-10k-C</th>
    </tr>
    <tr>
        <td>Tracker</td> <td>EAO</td> <td>A </td> <td>R</td><td>EAO&darr; <td>A&darr;<td>R&darr;</td><td>SR</td> <td>AO</td> <td>SR&darr;</td> <td>AO&darr;</td>
    </tr>

<tr>
<td>SiamRPN++                  </td> <td>23.7  </td><td>43.5  </td><td>63.5 </td><td>20.0(15.6)   </td><td>40.7(6.43)  </td><td>56.8(10.5) </td><td>80.7 </td><td>69.4  </td><td>75.5(6.44)  </td><td>64.7(6.77)</td>
</tr>
<tr>
<td>SiamMask~\cite{siammask}   </td> <td> 21.6 </td><td>43.7  </td><td>58.3 </td><td>19.1(11.6)   </td><td>42.3(3.20)  </td><td>51.8(11.1) </td><td> 75.3</td><td> 63.6 </td><td>72.7(3.45)  </td><td>61.9(2.67)</td>
</tr>
<tr>
<td>ECO~\cite{eco}             </td> <td> 27.9 </td><td>45.2  </td><td>74.3 </td><td> 25.3(9.32)  </td><td>44.0(2.65)  </td><td>69.3(6.73) </td><td>67.0 </td><td>57.9  </td><td>65.7(1.94)  </td><td>56.3(2.76)</td>
</tr>
<tr>
<td>DiMP-18~\cite{dimp}        </td> <td> 26.3 </td><td> 44.1 </td><td>70.0 </td><td> 23.0(12.5)  </td><td>41.6(5.67)  </td><td>63.0(10.0) </td><td>84.1 </td><td>71.6  </td><td>81.0 (3.68) </td><td>67.9(5.16</td>
</tr>
<tr>
<td>DiMP-50~\cite{dimp}        </td> <td> 26.4 </td><td> 43.6 </td><td>70.1 </td><td> 23.3(13.3)  </td><td>41.3(5.28)  </td><td>64.3(8.27) </td><td>86.7 </td><td>73.1  </td><td>81.1(6.46)  </td><td>68.5(6.29)</td>
</tr>
<tr>
<td>PrDiMP-18~\cite{prdimp}    </td> <td> 25.3 </td><td>45.7  </td><td>66.5 </td><td>22.5(11.1)   </td><td>44.0(3.72)  </td><td>59.6(10.4) </td><td>86.7 </td><td>74.4  </td><td>79.8(7.95)  </td><td>68.2(8.33)</td>
</tr>
<tr>
<td>PrDiMP-50~\cite{prdimp}    </td> <td> 26.6 </td><td>46.2  </td><td>68.8 </td><td>21.5(19.2)   </td><td>43.5(5.84)  </td><td>58.1(15.6) </td><td>87.5 </td><td>75.7  </td><td>81.5(6.86)  </td><td>69.5(8.19)</td>
</tr>
<tr>
<td>KeepTrack~\cite{keeptrack} </td> <td> 28.2 </td><td>46.6  </td><td>73.2 </td><td>23.8(15.6)   </td><td>43.9(5.79)  </td><td>64.3(12.2) </td><td>89.1 </td><td>77.4  </td><td>82.9(6.96)  </td><td>71.1(8.14)</td>
</tr>
<tr>
<td>KYS~\cite{kys}             </td> <td>26.0  </td><td>44.0  </td><td>68.6 </td><td>22.5(13.5)   </td><td>41.4(5.91)  </td><td>62.8(8.45) </td><td>87.1 </td><td>73.6  </td><td>81.4(6.54)  </td><td>68.7(6.66)</td>
</tr>
<tr>
<td>ToMP-50~\cite{tomp}        </td> <td>30.0  </td><td>45.3  </td><td>77.8 </td><td> 24.5(18.3)  </td><td>42.5(6.18)  </td><td>67.0(13.9) </td><td>93.2 </td><td>81.5  </td><td>87.6(6.01)  </td><td>75.3(7.61)</td>
</tr>
<tr>
<td>ToMP-101                   </td> <td>29.2  </td><td>44.9  </td><td>76.0 </td><td> 24.4(16.4)  </td><td> 43.0(4.23) </td><td>64.4(15.3) </td><td>91.9 </td><td>80.0  </td><td>85.8(6.63)  </td><td>73.3(8.37)</td>
</tr>
<tr>
<td>STARK-ST50~\cite{stark}    </td> <td>30.9  </td><td>46.9  </td><td>78.4 </td><td>24.4(21.0)   </td><td>43.6(7.04)  </td><td>65.4(16.6) </td><td>92.6 </td><td>81.4  </td><td>85.1(8.10)  </td><td>74.2(8.85)</td>
</tr>
<tr>
<td>STARK-ST101~\cite{stark}   </td> <td>30.7  </td><td>47.9  </td><td>75.4 </td><td>23.2(24.4)   </td><td>46.0(3.97)  </td><td>60.1(20.3) </td><td>92.6 </td><td>81.4  </td><td>85.2(7.99)  </td><td>74.0(9.10)</td>
</tr>
<tr>
<td>MixFormer~\cite{mixformer} </td> <td> 53.5 </td><td>76.2  </td><td>82.6 </td><td>42.6(20.4)   </td><td>65.5(14.0)  </td><td>76.1(7.87) </td><td>93.3 </td><td>82.7  </td><td>87.0(6.75)  </td><td>76.3(7.74)</td>
</tr>
<tr>
<td>MixFormer-Large            </td> <td>56.6  </td><td>76.8  </td><td>84.4 </td><td>44.7(21.0)   </td><td>67.0(12.8)  </td><td>77.7(7.94)
</tr>

</table>
