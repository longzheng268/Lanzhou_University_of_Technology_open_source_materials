-- Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
-- --------------------------------------------------------------------------------
-- Tool Version: Vivado v.2022.2 (win64) Build 3671981 Fri Oct 14 05:00:03 MDT 2022
-- Date        : Mon Oct 28 16:04:52 2024
-- Host        : cop running 64-bit major release  (build 9200)
-- Command     : write_vhdl -force -mode funcsim -rename_top decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix -prefix
--               decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_ risc32_MUX_20b_2_to_1_0_0_sim_netlist.vhdl
-- Design      : risc32_MUX_20b_2_to_1_0_0
-- Purpose     : This VHDL netlist is a functional simulation representation of the design and should not be modified or
--               synthesized. This netlist cannot be used for SDF annotated simulation.
-- Device      : xc7a100tcsg324-1
-- --------------------------------------------------------------------------------
library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
library UNISIM;
use UNISIM.VCOMPONENTS.ALL;
entity decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_MUX_20b_2_to_1 is
  port (
    result : out STD_LOGIC_VECTOR ( 19 downto 0 );
    data1 : in STD_LOGIC_VECTOR ( 19 downto 0 );
    data0 : in STD_LOGIC_VECTOR ( 19 downto 0 );
    sel : in STD_LOGIC
  );
end decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_MUX_20b_2_to_1;

architecture STRUCTURE of decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_MUX_20b_2_to_1 is
  attribute SOFT_HLUTNM : string;
  attribute SOFT_HLUTNM of \result[0]_INST_0\ : label is "soft_lutpair0";
  attribute SOFT_HLUTNM of \result[10]_INST_0\ : label is "soft_lutpair5";
  attribute SOFT_HLUTNM of \result[11]_INST_0\ : label is "soft_lutpair5";
  attribute SOFT_HLUTNM of \result[12]_INST_0\ : label is "soft_lutpair6";
  attribute SOFT_HLUTNM of \result[13]_INST_0\ : label is "soft_lutpair6";
  attribute SOFT_HLUTNM of \result[14]_INST_0\ : label is "soft_lutpair7";
  attribute SOFT_HLUTNM of \result[15]_INST_0\ : label is "soft_lutpair7";
  attribute SOFT_HLUTNM of \result[16]_INST_0\ : label is "soft_lutpair8";
  attribute SOFT_HLUTNM of \result[17]_INST_0\ : label is "soft_lutpair8";
  attribute SOFT_HLUTNM of \result[18]_INST_0\ : label is "soft_lutpair9";
  attribute SOFT_HLUTNM of \result[19]_INST_0\ : label is "soft_lutpair9";
  attribute SOFT_HLUTNM of \result[1]_INST_0\ : label is "soft_lutpair0";
  attribute SOFT_HLUTNM of \result[2]_INST_0\ : label is "soft_lutpair1";
  attribute SOFT_HLUTNM of \result[3]_INST_0\ : label is "soft_lutpair1";
  attribute SOFT_HLUTNM of \result[4]_INST_0\ : label is "soft_lutpair2";
  attribute SOFT_HLUTNM of \result[5]_INST_0\ : label is "soft_lutpair2";
  attribute SOFT_HLUTNM of \result[6]_INST_0\ : label is "soft_lutpair3";
  attribute SOFT_HLUTNM of \result[7]_INST_0\ : label is "soft_lutpair3";
  attribute SOFT_HLUTNM of \result[8]_INST_0\ : label is "soft_lutpair4";
  attribute SOFT_HLUTNM of \result[9]_INST_0\ : label is "soft_lutpair4";
begin
\result[0]_INST_0\: unisim.vcomponents.LUT3
    generic map(
      INIT => X"AC"
    )
        port map (
      I0 => data1(0),
      I1 => data0(0),
      I2 => sel,
      O => result(0)
    );
\result[10]_INST_0\: unisim.vcomponents.LUT3
    generic map(
      INIT => X"AC"
    )
        port map (
      I0 => data1(10),
      I1 => data0(10),
      I2 => sel,
      O => result(10)
    );
\result[11]_INST_0\: unisim.vcomponents.LUT3
    generic map(
      INIT => X"AC"
    )
        port map (
      I0 => data1(11),
      I1 => data0(11),
      I2 => sel,
      O => result(11)
    );
\result[12]_INST_0\: unisim.vcomponents.LUT3
    generic map(
      INIT => X"AC"
    )
        port map (
      I0 => data1(12),
      I1 => data0(12),
      I2 => sel,
      O => result(12)
    );
\result[13]_INST_0\: unisim.vcomponents.LUT3
    generic map(
      INIT => X"AC"
    )
        port map (
      I0 => data1(13),
      I1 => data0(13),
      I2 => sel,
      O => result(13)
    );
\result[14]_INST_0\: unisim.vcomponents.LUT3
    generic map(
      INIT => X"AC"
    )
        port map (
      I0 => data1(14),
      I1 => data0(14),
      I2 => sel,
      O => result(14)
    );
\result[15]_INST_0\: unisim.vcomponents.LUT3
    generic map(
      INIT => X"AC"
    )
        port map (
      I0 => data1(15),
      I1 => data0(15),
      I2 => sel,
      O => result(15)
    );
\result[16]_INST_0\: unisim.vcomponents.LUT3
    generic map(
      INIT => X"AC"
    )
        port map (
      I0 => data1(16),
      I1 => data0(16),
      I2 => sel,
      O => result(16)
    );
\result[17]_INST_0\: unisim.vcomponents.LUT3
    generic map(
      INIT => X"AC"
    )
        port map (
      I0 => data1(17),
      I1 => data0(17),
      I2 => sel,
      O => result(17)
    );
\result[18]_INST_0\: unisim.vcomponents.LUT3
    generic map(
      INIT => X"AC"
    )
        port map (
      I0 => data1(18),
      I1 => data0(18),
      I2 => sel,
      O => result(18)
    );
\result[19]_INST_0\: unisim.vcomponents.LUT3
    generic map(
      INIT => X"AC"
    )
        port map (
      I0 => data1(19),
      I1 => data0(19),
      I2 => sel,
      O => result(19)
    );
\result[1]_INST_0\: unisim.vcomponents.LUT3
    generic map(
      INIT => X"AC"
    )
        port map (
      I0 => data1(1),
      I1 => data0(1),
      I2 => sel,
      O => result(1)
    );
\result[2]_INST_0\: unisim.vcomponents.LUT3
    generic map(
      INIT => X"AC"
    )
        port map (
      I0 => data1(2),
      I1 => data0(2),
      I2 => sel,
      O => result(2)
    );
\result[3]_INST_0\: unisim.vcomponents.LUT3
    generic map(
      INIT => X"AC"
    )
        port map (
      I0 => data1(3),
      I1 => data0(3),
      I2 => sel,
      O => result(3)
    );
\result[4]_INST_0\: unisim.vcomponents.LUT3
    generic map(
      INIT => X"AC"
    )
        port map (
      I0 => data1(4),
      I1 => data0(4),
      I2 => sel,
      O => result(4)
    );
\result[5]_INST_0\: unisim.vcomponents.LUT3
    generic map(
      INIT => X"AC"
    )
        port map (
      I0 => data1(5),
      I1 => data0(5),
      I2 => sel,
      O => result(5)
    );
\result[6]_INST_0\: unisim.vcomponents.LUT3
    generic map(
      INIT => X"AC"
    )
        port map (
      I0 => data1(6),
      I1 => data0(6),
      I2 => sel,
      O => result(6)
    );
\result[7]_INST_0\: unisim.vcomponents.LUT3
    generic map(
      INIT => X"AC"
    )
        port map (
      I0 => data1(7),
      I1 => data0(7),
      I2 => sel,
      O => result(7)
    );
\result[8]_INST_0\: unisim.vcomponents.LUT3
    generic map(
      INIT => X"AC"
    )
        port map (
      I0 => data1(8),
      I1 => data0(8),
      I2 => sel,
      O => result(8)
    );
\result[9]_INST_0\: unisim.vcomponents.LUT3
    generic map(
      INIT => X"AC"
    )
        port map (
      I0 => data1(9),
      I1 => data0(9),
      I2 => sel,
      O => result(9)
    );
end STRUCTURE;
library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
library UNISIM;
use UNISIM.VCOMPONENTS.ALL;
entity decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix is
  port (
    data0 : in STD_LOGIC_VECTOR ( 19 downto 0 );
    data1 : in STD_LOGIC_VECTOR ( 19 downto 0 );
    sel : in STD_LOGIC;
    result : out STD_LOGIC_VECTOR ( 19 downto 0 )
  );
  attribute NotValidForBitStream : boolean;
  attribute NotValidForBitStream of decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix : entity is true;
  attribute CHECK_LICENSE_TYPE : string;
  attribute CHECK_LICENSE_TYPE of decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix : entity is "risc32_MUX_20b_2_to_1_0_0,MUX_20b_2_to_1,{}";
  attribute DowngradeIPIdentifiedWarnings : string;
  attribute DowngradeIPIdentifiedWarnings of decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix : entity is "yes";
  attribute IP_DEFINITION_SOURCE : string;
  attribute IP_DEFINITION_SOURCE of decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix : entity is "module_ref";
  attribute X_CORE_INFO : string;
  attribute X_CORE_INFO of decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix : entity is "MUX_20b_2_to_1,Vivado 2022.2";
end decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix;

architecture STRUCTURE of decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix is
begin
inst: entity work.decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_MUX_20b_2_to_1
     port map (
      data0(19 downto 0) => data0(19 downto 0),
      data1(19 downto 0) => data1(19 downto 0),
      result(19 downto 0) => result(19 downto 0),
      sel => sel
    );
end STRUCTURE;
