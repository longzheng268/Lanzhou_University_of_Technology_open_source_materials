-- Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
-- --------------------------------------------------------------------------------
-- Tool Version: Vivado v.2022.2 (win64) Build 3671981 Fri Oct 14 05:00:03 MDT 2022
-- Date        : Mon Oct 28 16:04:52 2024
-- Host        : cop running 64-bit major release  (build 9200)
-- Command     : write_vhdl -force -mode funcsim -rename_top decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix -prefix
--               decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_ risc32_MUX_5b_2_to_1_0_0_sim_netlist.vhdl
-- Design      : risc32_MUX_5b_2_to_1_0_0
-- Purpose     : This VHDL netlist is a functional simulation representation of the design and should not be modified or
--               synthesized. This netlist cannot be used for SDF annotated simulation.
-- Device      : xc7a100tcsg324-1
-- --------------------------------------------------------------------------------
library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
library UNISIM;
use UNISIM.VCOMPONENTS.ALL;
entity decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_MUX_5b_2_to_1 is
  port (
    result : out STD_LOGIC_VECTOR ( 4 downto 0 );
    data1 : in STD_LOGIC_VECTOR ( 4 downto 0 );
    sel : in STD_LOGIC;
    data0 : in STD_LOGIC_VECTOR ( 4 downto 0 )
  );
end decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_MUX_5b_2_to_1;

architecture STRUCTURE of decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_MUX_5b_2_to_1 is
  attribute SOFT_HLUTNM : string;
  attribute SOFT_HLUTNM of \result[0]_INST_0\ : label is "soft_lutpair0";
  attribute SOFT_HLUTNM of \result[1]_INST_0\ : label is "soft_lutpair0";
  attribute SOFT_HLUTNM of \result[2]_INST_0\ : label is "soft_lutpair1";
  attribute SOFT_HLUTNM of \result[3]_INST_0\ : label is "soft_lutpair1";
begin
\result[0]_INST_0\: unisim.vcomponents.LUT3
    generic map(
      INIT => X"B8"
    )
        port map (
      I0 => data1(0),
      I1 => sel,
      I2 => data0(0),
      O => result(0)
    );
\result[1]_INST_0\: unisim.vcomponents.LUT3
    generic map(
      INIT => X"B8"
    )
        port map (
      I0 => data1(1),
      I1 => sel,
      I2 => data0(1),
      O => result(1)
    );
\result[2]_INST_0\: unisim.vcomponents.LUT3
    generic map(
      INIT => X"B8"
    )
        port map (
      I0 => data1(2),
      I1 => sel,
      I2 => data0(2),
      O => result(2)
    );
\result[3]_INST_0\: unisim.vcomponents.LUT3
    generic map(
      INIT => X"B8"
    )
        port map (
      I0 => data1(3),
      I1 => sel,
      I2 => data0(3),
      O => result(3)
    );
\result[4]_INST_0\: unisim.vcomponents.LUT3
    generic map(
      INIT => X"B8"
    )
        port map (
      I0 => data1(4),
      I1 => sel,
      I2 => data0(4),
      O => result(4)
    );
end STRUCTURE;
library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
library UNISIM;
use UNISIM.VCOMPONENTS.ALL;
entity decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix is
  port (
    data0 : in STD_LOGIC_VECTOR ( 4 downto 0 );
    data1 : in STD_LOGIC_VECTOR ( 4 downto 0 );
    sel : in STD_LOGIC;
    result : out STD_LOGIC_VECTOR ( 4 downto 0 )
  );
  attribute NotValidForBitStream : boolean;
  attribute NotValidForBitStream of decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix : entity is true;
  attribute CHECK_LICENSE_TYPE : string;
  attribute CHECK_LICENSE_TYPE of decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix : entity is "risc32_MUX_5b_2_to_1_0_0,MUX_5b_2_to_1,{}";
  attribute DowngradeIPIdentifiedWarnings : string;
  attribute DowngradeIPIdentifiedWarnings of decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix : entity is "yes";
  attribute IP_DEFINITION_SOURCE : string;
  attribute IP_DEFINITION_SOURCE of decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix : entity is "module_ref";
  attribute X_CORE_INFO : string;
  attribute X_CORE_INFO of decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix : entity is "MUX_5b_2_to_1,Vivado 2022.2";
end decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix;

architecture STRUCTURE of decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix is
begin
inst: entity work.decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_MUX_5b_2_to_1
     port map (
      data0(4 downto 0) => data0(4 downto 0),
      data1(4 downto 0) => data1(4 downto 0),
      result(4 downto 0) => result(4 downto 0),
      sel => sel
    );
end STRUCTURE;
