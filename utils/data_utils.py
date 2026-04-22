from pathlib import Path

import pandas as pd


class Data_utils:
    """
    A utility class for handling data file operations.

    This class provides methods for reading and parsing data files with headers,
    particularly for OpenSim output files (.sto, .mot formats).
    """

    _skiprows_cache: dict[str, int] = {}

    def __init__(self):
        """Initialize the Data_utils instance."""

    def create_df(self, file_path):
        """
        Create a pandas DataFrame from a tab-separated data file.

        Args:
            file_path: Path to the data file

        Returns:
            pandas DataFrame with the file contents

        """
        # Read the file, skipping header rows until "endheader" marker
        df = pd.read_csv(
            file_path,
            sep="\t",
            skiprows=self.get_skiprows_count(file_path),
        )
        return df

    def get_skiprows_count(self, file_path):
        """
        Count the number of header rows to skip when reading a file.

        Args:
            file_path: Path to the data file

        Returns:
            Number of rows to skip (rows before and including "endheader")

        """
        resolved_path = str(Path(file_path).resolve())
        cached_skiprows = self._skiprows_cache.get(resolved_path)
        if cached_skiprows is not None:
            return cached_skiprows

        # Open and read the file
        with open(file_path) as file:
            lines = file.readlines()
            # Search for the "endheader" marker
            for i, line in enumerate(lines):
                if line.strip().startswith("endheader"):
                    # Return the line number after "endheader"
                    skiprows_count = i + 1
                    self._skiprows_cache[resolved_path] = skiprows_count
                    return skiprows_count
            # If no "endheader" found, don't skip any rows
            self._skiprows_cache[resolved_path] = 0
            return 0

    def get_skiprows_content(self, file_path):
        """
        Extract the header content from a file.

        Args:
            file_path: Path to the data file

        Returns:
            List of header lines (up to and including "endheader")

        """
        # Open and read the file
        with open(file_path) as file:
            lines = file.readlines()
            header_content = []
            # Collect all lines until "endheader" is found
            for line in lines:
                header_content.append(line.rstrip("\n"))
                if line.strip().startswith("endheader"):
                    break
            return header_content
        return []

    @staticmethod
    def trc_header_mod(data_input, new_line=None):
        """
        Modify the header of a TRC (Track Row Column) file.

        Args:
            data_input: Either a DataFrame or path to a TRC file
            new_line: Custom header line (optional, uses default if None)

        Returns:
            Modified DataFrame with updated header

        """
        # Set default header line if not provided
        if new_line is None:
            new_line = (
                "DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames"
            )

        # Handle input as DataFrame or file path
        if isinstance(data_input, pd.DataFrame):
            df = data_input
        else:
            file = str(data_input)
            # Validate that it's a TRC file
            if "trc" in file:
                df = pd.read_csv(file, header=None)
            else:
                raise ValueError("File is not a trc file")

        # Find the row containing "DataRate" and replace it
        for index, row in df.iterrows():
            if any("DataRate" in str(x) for x in row):
                df.iloc[index, 0] = new_line  # type: ignore
                index = index + 1  # type: ignore
                break

        # Modify the line below the header (values line)
        value_line = df.iloc[index, 0]  # type: ignore
        # Split by tab separator
        value_line = value_line.split("\t")
        # Get the value at the current index
        value = value_line[index]  # type: ignore
        # Append the value to the line
        value_line.append(value)
        # Rejoin and replace the old line
        df.iloc[index, 0] = "\t".join(value_line)  # type: ignore
        # Return the modified DataFrame
        return df
