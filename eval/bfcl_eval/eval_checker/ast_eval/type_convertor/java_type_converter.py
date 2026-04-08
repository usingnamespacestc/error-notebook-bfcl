import re
from typing import List, Dict, Union
from bfcl_eval.constants.type_mappings import JAVA_TYPE_CONVERSION


def java_type_converter(value, expected_type, nested_type=None):
    if expected_type not in JAVA_TYPE_CONVERSION:
        raise ValueError(f"Unsupported type: {expected_type}")
    if (
        expected_type == "byte"
        or expected_type == "short"
        or expected_type == "integer"
    ):
        if not re.match(r"^-?\d+$", value):
            return str(value)  # default to string
        return int(value)
    elif expected_type == "float":
        if not re.match(r"^-?\d+(\.\d+)?([eE][+-]?\d+)?[fF]$", value):
            return str(value)  # default to string
        return float(re.sub(r"[fF]$", "", value))
    elif expected_type == "double":
        if not re.match(r"^-?\d+(\.\d+)?([eE][+-]?\d+)?$", value):
            return str(value)  # default to string
        return float(value)
    elif expected_type == "long":
        if not re.match(r"^-?\d+[lL]$", value):
            return str(value)  # default to string
        return int(re.sub(r"[lL]$", "", value))
    elif expected_type == "boolean":
        if value not in ["true", "false"]:
            return str(value)  # default to string
        return value == "true"
    elif expected_type == "char":
        if not re.match(r"^\'.\'$", value):
            return str(value)  # default to string
        return value  # Remove the single quotes
    elif expected_type == "Array" or expected_type == "ArrayList":
        return parse_java_collection(value, expected_type, nested_type)
    elif expected_type == "Set":
        raise NotImplementedError("Set conversion is not implemented")
    elif expected_type == "HashMap":
        return parse_java_collection(value, expected_type, nested_type)
    elif expected_type == "Hashtable":
        raise NotImplementedError("Set conversion is not implemented")
    elif expected_type == "Queue" or expected_type == "Stack":
        raise NotImplementedError(f"{expected_type} conversion is not implemented")
    elif expected_type == "String" or expected_type == "any":
        return str(value)  # we output as string for `any` type
    else:
        raise ValueError(f"Unsupported type: {expected_type}")


def parse_java_collection(
    input_str: str, type_str: str, nested_type=None
) -> Union[List, Dict]:
    if type_str == "ArrayList":
        return parse_arraylist(input_str, nested_type)
    elif type_str == "Array":
        return parse_array(input_str, nested_type)
    elif type_str == "HashMap":
        return parse_hashmap(input_str)
    else:
        raise ValueError(f"Unsupported type: {type_str}")


def parse_arraylist(input_str: str, nested_type=None) -> List:
    match_asList = re.search(
        r"new\s+ArrayList<\w*>\(Arrays\.asList\((.+?)\)\)", input_str
    )
    if match_asList:
        elements_str = match_asList.group(1)
        elements = []
        for element_str in elements_str.split(","):
            element_str = element_str.strip()
            if nested_type == "char":
                element = element_str[1:-1]  # Remove the single quotes
            elif nested_type == "String":
                element = element_str[1:-1]  # Remove the double quotes
            else:
                element = (
                    java_type_converter(element_str, nested_type)
                    if nested_type
                    else parse_java_value(element_str)
                )
            elements.append(element)
        return elements

    match_add = re.search(
        r"new\s+ArrayList<\w*>\(\)\s*\{\{\s*(.+?)\s*\}\}", input_str, re.DOTALL
    )
    if match_add:
        adds_str = match_add.group(1)
        elements = []
        matches = re.findall(r"add\((.+?)\)", adds_str)
        for match in matches:
            value_str = match.strip()
            if nested_type == "char":
                value = value_str[1:-1]  # Remove the single quotes
            elif nested_type == "String":
                value = value_str[1:-1]  # Remove the double quotes
            else:
                value = (
                    java_type_converter(value_str, nested_type)
                    if nested_type
                    else parse_java_value(value_str)
                )
            elements.append(value)
        return elements

    match_empty = re.search(r"new\s+ArrayList<\w*>\(\)", input_str)
    if match_empty:
        return []  # Return an empty list for an empty ArrayList

    return input_str  # default to string


def parse_array(input_str: str, nested_type=None) -> List:
    match = re.search(r"new\s+\w+\[\]\s*\{(.*?)\}", input_str)
    if match:
        elements_str = match.group(1)
        if nested_type:
            elements = [
                java_type_converter(x.strip(), nested_type)
                for x in elements_str.split(",")
                if x.strip()
            ]
        else:
            elements = [
                parse_java_value(x.strip())
                for x in elements_str.split(",")
                if x.strip()
            ]

        return elements
    else:
        return input_str  # default to string


def parse_hashmap(input_str: str) -> Dict:
    elements = {}
    match = re.search(
        r"new\s+HashMap<.*?>\s*\(\)\s*\{\s*\{?\s*(.*?)\s*\}?\s*\}", input_str, re.DOTALL
    )
    if match:
        puts_str = match.group(1)
        if puts_str.strip():
            matches = re.findall(r'put\("(.*?)"\s*,\s*(.*?)\)', puts_str)
            for match in matches:
                key = match[0]
                value = parse_java_value(match[1].strip())
                elements[key] = value
        return elements

    match_empty = re.search(r"new\s+HashMap<.*?>\s*\(\)", input_str)
    if match_empty:
        return {}  # Return an empty dictionary for an empty HashMap

    return input_str  # default to string


def parse_java_value(value_str: str):
    # check if it's boolean
    if value_str == "true":
        return True
    elif value_str == "false":
        return False
    # check if it's a string
    elif value_str.startswith('"') and value_str.endswith('"'):
        return value_str[1:-1]
    # check if it's a long
    elif re.match(r"^-?\d+[lL]$", value_str):
        return int(value_str[:-1])
    # check if it's a float
    elif re.match(r"^-?\d+(\.\d+)?([eE][+-]?\d+)?[fF]$", value_str):
        return float(re.sub(r"[fF]$", "", value_str))
    # check if it's a integer-like and float-like types (including byte, short, integer, double, etc)
    else:
        try:
            return int(value_str)
        except ValueError:
            try:
                return float(value_str)
            except ValueError:
                # this assuming all other types are converted to string
                return value_str
