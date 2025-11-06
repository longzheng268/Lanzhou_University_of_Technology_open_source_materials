-- Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
-- --------------------------------------------------------------------------------
-- Tool Version: Vivado v.2022.2 (win64) Build 3671981 Fri Oct 14 05:00:03 MDT 2022
-- Date        : Tue Oct 29 10:56:53 2024
-- Host        : cop running 64-bit major release  (build 9200)
-- Command     : write_vhdl -force -mode funcsim -rename_top decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix -prefix
--               decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_ risc32_imem_0_0_sim_netlist.vhdl
-- Design      : risc32_imem_0_0
-- Purpose     : This VHDL netlist is a functional simulation representation of the design and should not be modified or
--               synthesized. This netlist cannot be used for SDF annotated simulation.
-- Device      : xc7a100tcsg324-1
-- --------------------------------------------------------------------------------
library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
library UNISIM;
use UNISIM.VCOMPONENTS.ALL;
entity decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_imem is
  port (
    inst_o : out STD_LOGIC_VECTOR ( 9 downto 0 );
    inst_addr : in STD_LOGIC_VECTOR ( 5 downto 0 )
  );
end decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_imem;

architecture STRUCTURE of decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_imem is
begin
\inst_o[10]_INST_0\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"0100000000000010"
    )
        port map (
      I0 => inst_addr(1),
      I1 => inst_addr(0),
      I2 => inst_addr(5),
      I3 => inst_addr(4),
      I4 => inst_addr(3),
      I5 => inst_addr(2),
      O => inst_o(2)
    );
\inst_o[15]_INST_0\: unisim.vcomponents.LUT5
    generic map(
      INIT => X"00000002"
    )
        port map (
      I0 => inst_addr(5),
      I1 => inst_addr(3),
      I2 => inst_addr(1),
      I3 => inst_addr(4),
      I4 => inst_addr(0),
      O => inst_o(4)
    );
\inst_o[20]_INST_0\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"0001000100011001"
    )
        port map (
      I0 => inst_addr(0),
      I1 => inst_addr(1),
      I2 => inst_addr(5),
      I3 => inst_addr(2),
      I4 => inst_addr(4),
      I5 => inst_addr(3),
      O => inst_o(5)
    );
\inst_o[21]_INST_0\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"0001000300000300"
    )
        port map (
      I0 => inst_addr(4),
      I1 => inst_addr(1),
      I2 => inst_addr(0),
      I3 => inst_addr(3),
      I4 => inst_addr(5),
      I5 => inst_addr(2),
      O => inst_o(6)
    );
\inst_o[23]_INST_0\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"0000100000100000"
    )
        port map (
      I0 => inst_addr(1),
      I1 => inst_addr(0),
      I2 => inst_addr(2),
      I3 => inst_addr(4),
      I4 => inst_addr(5),
      I5 => inst_addr(3),
      O => inst_o(7)
    );
\inst_o[24]_INST_0\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"0000000000000100"
    )
        port map (
      I0 => inst_addr(0),
      I1 => inst_addr(4),
      I2 => inst_addr(1),
      I3 => inst_addr(5),
      I4 => inst_addr(3),
      I5 => inst_addr(2),
      O => inst_o(8)
    );
\inst_o[31]_INST_0\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"0000000000000001"
    )
        port map (
      I0 => inst_addr(5),
      I1 => inst_addr(4),
      I2 => inst_addr(1),
      I3 => inst_addr(0),
      I4 => inst_addr(3),
      I5 => inst_addr(2),
      O => inst_o(9)
    );
\inst_o[5]_INST_0\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"0000000000001000"
    )
        port map (
      I0 => inst_addr(1),
      I1 => inst_addr(4),
      I2 => inst_addr(5),
      I3 => inst_addr(2),
      I4 => inst_addr(3),
      I5 => inst_addr(0),
      O => inst_o(3)
    );
\inst_o[7]_INST_0\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"0000000001010111"
    )
        port map (
      I0 => inst_addr(0),
      I1 => inst_addr(2),
      I2 => inst_addr(5),
      I3 => inst_addr(3),
      I4 => inst_addr(4),
      I5 => inst_addr(1),
      O => inst_o(0)
    );
\inst_o[8]_INST_0\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"0000010101010010"
    )
        port map (
      I0 => inst_addr(0),
      I1 => inst_addr(1),
      I2 => inst_addr(5),
      I3 => inst_addr(4),
      I4 => inst_addr(3),
      I5 => inst_addr(2),
      O => inst_o(1)
    );
end STRUCTURE;
library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
library UNISIM;
use UNISIM.VCOMPONENTS.ALL;
entity decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix is
  port (
    inst_addr : in STD_LOGIC_VECTOR ( 31 downto 0 );
    inst_o : out STD_LOGIC_VECTOR ( 31 downto 0 )
  );
  attribute NotValidForBitStream : boolean;
  attribute NotValidForBitStream of decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix : entity is true;
  attribute CHECK_LICENSE_TYPE : string;
  attribute CHECK_LICENSE_TYPE of decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix : entity is "risc32_imem_0_0,imem,{}";
  attribute DowngradeIPIdentifiedWarnings : string;
  attribute DowngradeIPIdentifiedWarnings of decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix : entity is "yes";
  attribute IP_DEFINITION_SOURCE : string;
  attribute IP_DEFINITION_SOURCE of decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix : entity is "module_ref";
  attribute X_CORE_INFO : string;
  attribute X_CORE_INFO of decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix : entity is "imem,Vivado 2022.2";
end decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix;

architecture STRUCTURE of decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix is
  signal \<const0>\ : STD_LOGIC;
  signal \^inst_o\ : STD_LOGIC_VECTOR ( 31 downto 4 );
begin
  inst_o(31) <= \^inst_o\(31);
  inst_o(30) <= \<const0>\;
  inst_o(29) <= \<const0>\;
  inst_o(28) <= \<const0>\;
  inst_o(27) <= \<const0>\;
  inst_o(26) <= \<const0>\;
  inst_o(25) <= \<const0>\;
  inst_o(24 downto 20) <= \^inst_o\(24 downto 20);
  inst_o(19) <= \<const0>\;
  inst_o(18) <= \<const0>\;
  inst_o(17) <= \<const0>\;
  inst_o(16) <= \<const0>\;
  inst_o(15) <= \^inst_o\(15);
  inst_o(14) <= \<const0>\;
  inst_o(13) <= \<const0>\;
  inst_o(12) <= \<const0>\;
  inst_o(11 downto 7) <= \^inst_o\(11 downto 7);
  inst_o(6) <= \<const0>\;
  inst_o(5) <= \^inst_o\(11);
  inst_o(4) <= \^inst_o\(4);
  inst_o(3) <= \<const0>\;
  inst_o(2) <= \<const0>\;
  inst_o(1) <= \^inst_o\(4);
  inst_o(0) <= \^inst_o\(4);
GND: unisim.vcomponents.GND
     port map (
      G => \<const0>\
    );
inst: entity work.decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_imem
     port map (
      inst_addr(5 downto 0) => inst_addr(5 downto 0),
      inst_o(9) => \^inst_o\(31),
      inst_o(8 downto 7) => \^inst_o\(24 downto 23),
      inst_o(6 downto 5) => \^inst_o\(21 downto 20),
      inst_o(4) => \^inst_o\(15),
      inst_o(3 downto 2) => \^inst_o\(11 downto 10),
      inst_o(1 downto 0) => \^inst_o\(8 downto 7)
    );
\inst_o[0]_INST_0\: unisim.vcomponents.LUT5
    generic map(
      INIT => X"0000001F"
    )
        port map (
      I0 => inst_addr(4),
      I1 => inst_addr(3),
      I2 => inst_addr(5),
      I3 => inst_addr(0),
      I4 => inst_addr(1),
      O => \^inst_o\(4)
    );
\inst_o[22]_INST_0\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"0000000000001464"
    )
        port map (
      I0 => inst_addr(5),
      I1 => inst_addr(4),
      I2 => inst_addr(2),
      I3 => inst_addr(3),
      I4 => inst_addr(0),
      I5 => inst_addr(1),
      O => \^inst_o\(22)
    );
\inst_o[9]_INST_0\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"0000000000001324"
    )
        port map (
      I0 => inst_addr(3),
      I1 => inst_addr(5),
      I2 => inst_addr(2),
      I3 => inst_addr(4),
      I4 => inst_addr(0),
      I5 => inst_addr(1),
      O => \^inst_o\(9)
    );
end STRUCTURE;
