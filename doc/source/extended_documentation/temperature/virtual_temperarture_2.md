## Virtual Temperature Calculations

Two virtual temperature calculations are currently provided:


### 1. VirtualTemperature Class Calculation

input diagnostics:

          1. temperature ( T )
          2. humidity_mixing_ratio ( q )

<br>

This IMPROVER plugin uses a first order approximation for virtual temperature ($T_v$)

You can show:

$$
T_{v} = T \left[ \frac{q}{\epsilon} + ( 1 - q) \right] =  T \left[ 1 + q\left( \frac{1}{\epsilon} - 1 \right) \right]
$$
Since
$$
\frac{1}{\epsilon} - 1 \approx 0.62
$$
Gives:
$$
T_v \approx T ( 1 + 0.62q )
$$

where **$\epsilon$** is the ratio of the gas constants (or molecular weights) of dry air and water vapour:



$$
\epsilon = \frac{R_d}{R_v}
$$

where:

- $R_d \approx 287.05 \,\mathrm{J\,kg^{-1}\,K^{-1}}$ is the gas constant for dry air.
- $R_v \approx 461.5 \,\mathrm{J\,kg^{-1}\,K^{-1}}$ is the gas constant for water vapour.

So:


$$
\epsilon \approx \frac{287.05}{461.5} \approx 0.622
$$

 
An equivalent definition using molecular weights is:
 
$$
\epsilon = \frac{M_v}{M_d}
$$

where:

- $M_v \approx 18.016 \,\mathrm{g\,mol^{-1}}$ (water vapour)
- $M_d \approx 28.964 \,\mathrm{g\,mol^{-1}}$ (dry air)

giving the same value:

$$
\epsilon \approx 0.622
$$

### 1. VirtualTemperatureFromSpecificHumidityClass Calculation

input diagnostics:

      1. temperature ( T )
      2. humidity_mixing_ratio ( q )
      3. cloud_water_mixing_ratio
      4. cloud_ice_mixing_ratio


<br>

This calculates virtual temperature using the specific humidity, which is desirable in the calculation of Air Density.

This virtual temperature ($T_v$) calculation also uses condensates, if provided.

Condensed water (rain, snow and liquid cloud droplets) add weight to a volume of air
without contributing to the pressure.
Their effect can be included in the calculation of virtual temperature.

When the data on the mixing ratios of liquid water ($q_{cl}$) and ice ($q_{cf}$) are available, a slightly improved estimate of Tv, accounting for this weight, is:

$$
    T_v = T \left[ \left(\frac{q}{\epsilon}\right) + (1 - q - q_{cl} - q_{cf}) \right]
$$

This is only significant lower down in the atmosphere, and where there is cloud,
and in most cases a very small adjustment. Given the densest clouds have condensate
specific humidities typically no higher than 5 g kg-1, the adjustment is unlikely
to ever be more than 0.5%.
The virtual temperature calculation can be modified if condensate information is available (Equation 4), which it is from StaGE. This can include condensate information on liquid water, ice, and rain. Subtracting this condensate moisture from within the virtual temperature calculation will have the effect of reducing density slightly.

where:

$T_v$ = virtual temperature (K)

$T$ = temperature (K)

$q$ = specific humidity (a measure of the actual amount of water vapour in the air, expressed as kg kg-1)

$\epsilon$ = 0.62198 is the ratio of gas constants of dry to moist air.

(See EARTH_REPSILON constant in IMPROVER.constants)

$q_{cl}$
 = mixing ratio of liquid water (StaGE input: cloud_water_mixing_ratio_on_pressure_levels - kg kg-1)

$q_{cf}$ = mixing ratio of ice (StaGE input: cloud_ice_mixing_ratio_on_pressure_levels - kg kg-1)

Excluding condensates from the virtual temperature calculation will reduce the amount of moisture and will therefore reduce density. This is only significant lower down in the atmosphere, and where there is cloud, and in most cases leads to a very small adjustment. Given the densest clouds have condensate specific humidities typically no higher than 5 g kg-1, the adjustment is unlikely to ever be more than 0.5%.
