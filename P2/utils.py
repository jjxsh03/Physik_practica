import pandas

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
