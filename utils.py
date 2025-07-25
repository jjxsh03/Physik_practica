import pandas
import matplotlib.pyplot as plt

# class for definition of Dataset used with pandas
class ReadSet:
    '''
    Utility Class to read different datasets
    use per:
    >reader = ReadSet()
    >df = reader.read
    '''
    def __init__(self):
        self.supported_types = ['csv', 'txt', 'dat', 'xlsx']

    def read(self, filename, filetype=None):
        if filetype is None:
            filetype = input('Enter data type (csv, txt, dat, xlsx): ').lower()
        else:
            filetype = filetype.lower()

        if not os.path.exists(filename):
            raise FileNotFoundError(f"File '{filename}' not found.")

        if filetype == 'csv':
            return pd.read_csv(filename)

        elif filetype in ['txt', 'dat']:
            return pd.read_csv(filename,sep = None)

        elif filetype == 'xlsx':
            return pd.read_excel(filename)

        else:
            raise ValueError(f"Unsupported file type: {filetype}")

# Plot Data does need to be changed to work, in Progress of change
class PlotData:
    def __init__(self):

    def forward():

        fig.ax = plt.subplots()

        ax.error() # Experimental Data and Error Data
        ax.plot(, ls = '--', color = 'red' , label = 'Model data') # Fit from Modelled Data
        ax.legend()
        ax.set_xlabel()
        ax.set_ylabel()
        ax.set_title()

        plt.savefig()
        plt.close()

        
