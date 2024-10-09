"""
Class for Digester Information.

The necessary information about the digester configuration for the ADM1 routine to work is the
liquid phase volume, gas phase volume and Temperature.

The digester information can be loaded from a json file using load_dig_info.
The digester information can be saved to a json file using the .save method.
"""
from vsr.basic_io import rw_jsonlike


class ADM1DigInfo:
    """
    Class for Digester Information.

    Attributes:
        V_liq, the volume of the liquid phase in M3
        V_gas, the volume of the gas phase in M3
    """

    def __init__(self, V_liq: float, V_gas: float):
        assert V_liq > 0, "The liquid phase volume must be strictly positive"
        assert V_gas > 0, "The gas phase volume must be strictly positive"

        self.V_liq = float(V_liq)
        self.V_gas = float(V_gas)

    @property
    def V_liq(self):
        """Liquid volume (in m3)"""
        return self._V_liq

    @V_liq.setter
    def V_liq(self, val):
        x = float(val)
        if x <= 0.0:
            raise ValueError(f"V_liq should be positive (passed {x})")
        self._V_liq = x

    @property
    def V_gas(self):
        """Gas volume (in m3)"""
        return self._V_gas

    @V_gas.setter
    def V_gas(self, val):
        x = float(val)
        if x <= 0.0:
            raise ValueError(f"V_gas should be positive (passed {x})")
        self._V_gas = x

    def save(self, path):
        """Save ADM1DigInfo object to .json file"""
        rw_jsonlike.save(
            path,
            {
                "V_liq": self.V_liq,
                "V_gas": self.V_gas,
            },
        )

    def __str__(self):
        return str.join(
            "\n",
            [
                f"V_liq: {self.V_liq}",
                f"V_gas: {self.V_gas}",
            ],
        )

    def __repr__(self):
        return str.join(
            "\n",
            [
                f"V_liq: {self.V_liq}",
                f"V_gas: {self.V_gas}",
            ],
        )


def load_dig_info(path: str) -> ADM1DigInfo:
    return ADM1DigInfo(**rw_jsonlike.load(path))
