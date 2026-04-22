import xml.etree.ElementTree as ET
from pathlib import Path


class XML_utils:
    """
    A utility class for XML file manipulation operations.

    This class provides methods for parsing, modifying, and writing XML files,
    commonly used for OpenSim setup and configuration files.
    """

    def __init__(self):
        """Initialize the XML_utils instance."""

    @staticmethod
    def xml_parse(xml_file_path: Path):
        """
        Parse an XML file and return the ElementTree object.

        Args:
            xml_file_path: Path to the XML file

        Returns:
            ElementTree object representing the parsed XML

        Raises:
            ValueError: If parsing fails or file is invalid

        """
        try:
            # Ensure path is a Path object
            xml_file_path = Path(xml_file_path)
            # Parse the XML file
            tree = ET.parse(xml_file_path)
            if tree is None:
                raise ValueError(f"Failed to parse XML file: {xml_file_path}")
            return tree
        except ET.ParseError as e:
            raise ValueError(f"Error parsing XML file: {e}")

    def xml_add_element(
        self,
        xml_file_path,
        parent_element_name: str,
        element_to_add: ET.Element,
    ) -> None:
        """
        Add an existing XML element to the tree under a specified parent element.

        Args:
            xml_file_path: Path to the XML file
            parent_element_name: Name/path of the parent element (supports XPath)
            element_to_add: XML Element object to add

        Raises:
            ValueError: If parent element is not found or other error occurs

        """
        try:
            # Parse the XML file to get the tree structure
            tree = XML_utils.xml_parse(xml_file_path)
            root = tree.getroot()
            if root is None:
                raise ValueError(f"Root element not found in {xml_file_path}.")

            # Find the parent element using XPath notation
            parent = root.find(f".//{parent_element_name}")
            if parent is None:
                raise ValueError(
                    f"Parent element '{parent_element_name}' not found in {xml_file_path}.",
                )

            # Append the new element to the parent
            parent.append(element_to_add)

            # Write the modified tree back to the file
            tree.write(xml_file_path, encoding="utf-8", xml_declaration=True)
        except Exception as e:
            raise ValueError(f"Error adding element: {e}")

    def xml_add_text(self, xml_file_path, element_name: str, text: str):
        """
        Add or modify text content of an existing XML element.

        Args:
            xml_file_path: Path to the XML file
            element_name: Name/path of the element to modify (supports XPath)
            text: Text content to set

        Raises:
            ValueError: If element is not found or other error occurs

        """
        try:
            # Parse the XML file to get the tree structure
            tree = XML_utils.xml_parse(xml_file_path)
            root = tree.getroot()
            if root is None:
                raise ValueError(f"Root element not found in {xml_file_path}.")

            # Find the target element using XPath notation
            element = root.find(f".//{element_name}")
            if element is None:
                raise ValueError(
                    f"Element '{element_name}' not found in {xml_file_path}.",
                )

            # Set the text content of the element
            element.text = text

            # Write the modified tree back to the file
            tree.write(xml_file_path, encoding="utf-8", xml_declaration=True)
        except Exception as e:
            raise ValueError(f"Error adding text: {e}")
