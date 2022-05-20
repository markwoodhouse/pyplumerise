from textwrap import wrap

def plotTitle(titleStr, linelength=60):
    """Format a long plot title, nicely wrapping text

    Args:
        titleStr (str): the title for the plot
        linelength (int, optional): Maximum line length for wrapping. Defaults to 60.

    Returns:
        str: The title with line breaks
    """
    return "\n".join(wrap(titleStr, linelength))
