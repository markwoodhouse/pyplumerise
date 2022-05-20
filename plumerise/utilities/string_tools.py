from pint import UnitRegistry

ureg = UnitRegistry()

ureg.define("km_a_v_l = km = km a.v.l.")
ureg.define("m_a_s_l = m = m a.s.l.")
ureg.define("wt_percent = [] = wt. %")
ureg.define("percent = [] = %")
ureg.define("phi_unit = [] = phi unit")
ureg.define("m3 = m**3 = m^3")


def string_to_pint(unit):

    unit = unit.replace("km a.v.l.", "km_a_v_l")
    unit = unit.replace("m a.s.l.", "m_a_s_l")
    unit = unit.replace("wt.%", "wt_percent")
    unit = unit.replace("%", "percent")
    unit = unit.replace("phi unit", "phi_unit")
    unit = unit.replace("m^3", "m3")

    brks = [p for p, char in enumerate(unit) if char in [" ", "*", "/"]]
    op = [unit[k] for k in brks]
    brks.append(len(unit))

    units = []
    k = 0
    for j in brks:
        units.append(unit[k:j])
        k = j + 1

    for j, u in enumerate(units):
        if u == "1":
            u = "dimensionless"
        if j == 0:
            U = ureg.parse_expression(u).units
        else:
            if op[j - 1] in [" ", "*"]:
                U *= ureg.parse_expression(u).units
            elif op[j - 1] == "/":
                U /= ureg.parse_expression(u).units

    return U


def units_html(unit):

    if unit is not None:

        U = string_to_pint(unit)

        ustr = "{:~H}".format(U)

        if ustr == "1/s":
            ustr = "s<sup>-1</sup>"
        if ustr == "m^3":
            ustr = "m<sup>3</sup>"
        if ustr == "km^3":
            ustr = "km<sup>3</sup>"
        return ustr
    else:
        return ""


def string_html(strIn):

    if len(strIn) < 16:
        return strIn

    splits = strIn.split()

    strOut = ""
    currentLine = ""
    for sp in splits:
        strOut += sp
        currentLine += sp
        if len(currentLine) > 10:
            strOut += "<br>"
            currentLine = ""
        else:
            strOut += " "

    return strOut
