import numpy as np
tables = [
    # "SiamRPN++~\cite{siamrpn++}          &23.7 &20.0 &69.4 &64.7 &41.4 &44.9 &52.5 &51.7",
    # "SiamMask~\cite{siammask}            &21.6 &19.1 &63.6 &61.9 &37.9 &27.8 &54.5 &46.2",
    # "ECO~\cite{eco}                      &27.9 &25.3 &57.9 &56.3 &31.9 &23.3 &44.8 &43.6",
    # "DiMP-50~\cite{dimp}                 &26.4 &23.3 &73.1 &68.5 &57.3 &43.3 &56.8 &53.3",
    # "PrDiMP-50~\cite{prdimp}             &26.6 &21.5 &75.7 &69.5 &57.3 &45.9 &68.3 &60.6",
    # "KeepTrack~\cite{keeptrack}          &28.2 &23.8 &77.4 &71.1 &70.1 &65.8 &69.2 &70.5",
    # "KYS~\cite{kys}                      &26.0 &22.5 &73.6 &68.7 &32.5 &25.3 &54.9 &52.0",
    # "ToMP-50~\cite{tomp}                 &30.0 &24.5 &81.5 &75.3 &67.2 &63.5 &66.1 &61.6",
 
    # "STARK-ST50~\cite{stark}             &61.0 &48.5 &81.4 &74.2 &89.3 &82.7 &74.3 &65.1"]
    "SiamRPN++~\cite{siamrpn++}          &42.6 &48.4 &69.4 &64.7 &41.4 &44.9 &52.5 &51.7",
    "SiamMask~\cite{siammask}            &41.0 &34.0 &63.6 &61.9 &37.9 &27.8 &54.5 &46.2",
    "ECO~\cite{eco}                      &44.6 &33.7 &57.9 &56.3 &31.9 &23.3 &44.8 &43.6",
    "DiMP-50~\cite{dimp}                 &55.9 &45.8 &73.1 &68.5 &57.3 &43.3 &56.8 &53.3",
    "PrDiMP-50~\cite{prdimp}             &56.3 &43.9 &75.7 &69.5 &57.3 &45.9 &68.3 &60.6",
    "KeepTrack~\cite{keeptrack}          &55.7 &48.3 &77.4 &71.1 &70.1 &65.8 &69.2 &70.5",
    "KYS~\cite{kys}                      &51.2 &39.0 &73.6 &68.7 &32.5 &25.3 &54.9 &52.0",
    "ToMP-50~\cite{tomp}                 &59.9 &57.8 &81.5 &75.3 &67.2 &63.5 &66.1 &61.6",
    "STARK-ST50~\cite{stark}             &61.0 &48.5 &81.4 &74.2 &89.3 &82.7 &74.3 &65.1"]
   
textdata = []
for tab in tables:
    text = tab.split("&")
    textdata.append(text[1:])

data = np.float16(textdata)

column_average = np.mean(data,0)

print("column_average: {}".format(column_average))
