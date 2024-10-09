class ADM1Fractionner:

    __gCOD = 350  # mlCH4/gVS
    __cod_XI = 1.6  # gCOD/gVS
    __N_XI = 0.005  # moleN/gCOD
    __N_prot = 0.0121  # moleN/gCOD

    @property
    def gCOD(self):
        """1g of COD in mLCH4/gVS"""
        return self.__gCOD

    @property
    def cod_XI(self):
        """in gCOD/gVS"""
        return self.__cod_XI

    @property
    def N_XI(self):
        """N in inert (in moleN/gCOD)"""
        return self.__N_XI

    @property
    def N_prot(self):
        """N in proteins (in moleN/gCOD)"""
        return self.__N_prot

    __lip_cod_vs = 2.0  # COD/VS
    __prot_cod_vs = 1.5  # COD/VS
    __carb_cod_vs = 1.03  # COD/VS

    @property
    def lip_cod_vs(self):
        """in COD/VS"""
        return self.__lip_cod_vs

    @property
    def prot_cod_vs(self):
        """in COD/VS"""
        return self.__prot_cod_vs

    @property
    def carb_cod_vs(self):
        """in COD/VS"""
        return self.__carb_cod_vs

    def __init__(self, ts: float, vs_frac: float, cod_vs, carb, prot, Bo, khyd):
        if (ts < 0.0) or (ts > 1.0):
            raise ValueError(
                "Total Solid not in valid range (should be between 0 and 1)"
            )

        if (vs_frac < 0.0) or (vs_frac > 1.0):
            raise ValueError(
                "Volatile Solid Fraction not in valid range (should be between 0 and 1)"
            )

        self.__ts = ts
        self.__vs_frac = vs_frac
        self.__cod_vs = cod_vs

        self.__carb = carb
        self.__prot = prot

        self.__Bo = Bo
        self.__khyd = khyd

        self._comp_vs()
        self._comp_lip()
        self._comp_degrad()

    def _comp_vs(self):
        self.__vs = self.__ts * self.__vs_frac

    def _comp_lip(self):
        self.__lip = 1 - self.__carb - self.__prot

    def _comp_degrad(self):
        self.__degrad = self.__Bo / (self.__cod_vs * self.__gCOD)

    @property
    def Bo(self):
        """BMP NmL/gVS"""
        return self.__Bo

    @Bo.setter
    def Bo(self, value):
        self.__Bo = value
        self._comp_degrad()

    @property
    def khyd(self):
        """Hydrolysis constant (D-1)"""
        return self.__khyd

    @khyd.setter
    def khyd(self, value):
        self.__khyd = value

    @property
    def cod_vs(self):
        return self.__cod_vs

    @cod_vs.setter
    def cod_vs(self, value):
        self.__cod_vs = value
        self._comp_degrad()

    @property
    def carb(self):
        return self.__carb

    @carb.setter
    def carb(self, value):
        """Carbohydrate fraction"""
        self.__carb = value
        self._comp_lip()

    @property
    def prot(self):
        """Protein fraction"""
        return self.__prot

    @prot.setter
    def prot(self, value):
        self.__prot = value
        self._comp_lip()

    @property
    def lip(self):
        """Lipid fraction"""
        return self.__lip

    @property
    def agg_cod_vs(self):
        agg = (
            self.__lip * self.__lip_cod_vs
            + self.__carb * self.__carb_cod_vs
            + self.__prot * self.__prot_cod_vs
        ) * self.__degrad

        agg += self.__cod_XI * (1 - self.__degrad)
        return agg

    @property
    def degrad(self):
        return self.__degrad

    @property
    def Bo_degrad(self):
        return self.__cod_vs * self.__gCOD

    @property
    def nitrogen_content(self):
        return self.__N_prot * self.__prot * self.__degrad + self.__N_XI * (
            1 - self.__degrad
        )

    @property
    def overall_n_content(self):
        return self.nitrogen_content / self.__cod_vs * 14

    @property
    def ts(self):
        """Total Solid (ratio)"""
        return self.__ts

    @ts.setter
    def ts(self, value):
        if (value < 0.0) or (value > 1.0):
            raise ValueError(
                "Total Solid not in valid range (should be between 0 and 1)"
            )

        self.__ts = value
        self._comp_vs()

    @property
    def vs_frac(self):
        """Fraction of volatile solid in total solid"""
        return self.__vs_frac

    @vs_frac.setter
    def vs_frac(self, value):
        self.__vs_frac = value
        self._comp_vs()

    @property
    def vs(self):
        """Volatile solid"""
        return self.__vs

    @property
    def Xch(self):
        """per gCOD"""
        return self.__degrad * self.__carb

    @property
    def Xli(self):
        """per gCOD"""
        return self.__degrad * self.__lip

    @property
    def Xpr(self):
        """per gCOD"""
        return self.__degrad * self.__prot

    @property
    def XI(self):
        """per gCOD"""
        return 1 - self.__degrad

    @property
    def Xs(self):
        """Decomposition of the VS in ch, pr, li and I"""
        return self.Xch, self.Xpr, self.Xli, self.XI


primary_sludge_fract = ADM1Fractionner(
    ts=1.0, vs_frac=0.8, cod_vs=1.8, carb=0.1, prot=0.1, khyd=0.5, Bo=450
)

biological_sludge_fract = ADM1Fractionner(
    ts=1.0,
    vs_frac=0.8,
    cod_vs=1.5,
    carb=0.4,
    prot=0.5,
    khyd=0.3,
    Bo=250,
)
