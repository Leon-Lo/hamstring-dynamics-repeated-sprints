import os
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from utils.xml_utils import XML_utils


class Test_xml_utils:
    @pytest.fixture
    def temp_xml_file(self, tmp_path: Path):
        # Create a temporary XML file for testing
        file_path = tmp_path / "test.xml"
        xml_content = """<?xml version="1.0"?>
        <root>
            <child>Test</child>
            <subchild>
                <subsubchild>SubTest</subsubchild>
            </subchild>
        </root>"""
        try:
            file_path.write_text(xml_content, encoding="utf-8")
            yield file_path
        finally:
            # Cleanup the temporary file
            if file_path.exists():
                os.remove(file_path)

    def test_xml_parse(self, temp_xml_file: Path):
        # Test parsing the XML file
        parse = XML_utils().xml_parse(temp_xml_file)
        assert isinstance(parse, ET.ElementTree)
        assert parse.getroot().tag == "root"

    def test_xml_add_element(self, temp_xml_file: Path):
        # Test adding an element to the XML file
        xml_processor = XML_utils()

        # Create a new element to add
        new_element = ET.Element("new_child")
        new_element.text = "NewTest"

        # Add the new element to the XML file
        xml_processor.xml_add_element(temp_xml_file, "child", new_element)

        modified_tree = xml_processor.xml_parse(temp_xml_file)
        modified_root = modified_tree.getroot()

        added_element = modified_root.find("./child/new_child")

        assert added_element is not None
        assert isinstance(added_element, ET.Element)
        assert added_element.tag == "new_child"
        assert added_element.text == "NewTest"

    def test_xml_add_text(self, temp_xml_file: Path):
        xml_processor = XML_utils()

        # Add text to an existing element
        xml_processor.xml_add_text(temp_xml_file, "child", "ModifiedText")
        modified_tree = xml_processor.xml_parse(temp_xml_file)
        modified_root = modified_tree.getroot()
        modified_element = modified_root.find("./child")
        if modified_element is None:
            raise ValueError("Element 'child' not found in the modified XML.")

        added_text = modified_element.text

        assert added_text == "ModifiedText"
        assert isinstance(added_text, str)
