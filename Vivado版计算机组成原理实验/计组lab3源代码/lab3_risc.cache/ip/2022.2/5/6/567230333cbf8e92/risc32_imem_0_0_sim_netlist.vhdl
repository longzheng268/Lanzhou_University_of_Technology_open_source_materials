-- Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
-- --------------------------------------------------------------------------------
-- Tool Version: Vivado v.2022.2 (win64) Build 3671981 Fri Oct 14 05:00:03 MDT 2022
-- Date        : Mon Oct 28 16:05:23 2024
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
    inst : out STD_LOGIC_VECTOR ( 9 downto 0 );
    inst_addr : in STD_LOGIC_VECTOR ( 5 downto 0 )
  );
end decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_imem;

architecture STRUCTURE of decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_imem is
begin
\inst[10]_INST_0\: unisim.vcomponents.LUT6
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
      O => inst(2)
    );
\inst[15]_INST_0\: unisim.vcomponents.LUT5
    generic map(
      INIT => X"00000002"
    )
        port map (
      I0 => inst_addr(5),
      I1 => inst_addr(3),
      I2 => inst_addr(1),
      I3 => inst_addr(4),
      I4 => inst_addr(0),
      O => inst(4)
    );
\inst[20]_INST_0\: unisim.vcomponents.LUT6
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
      O => inst(5)
    );
\inst[21]_INST_0\: unisim.vcomponents.LUT6
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
      O => inst(6)
    );
\inst[23]_INST_0\: unisim.vcomponents.LUT6
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
      O => inst(7)
    );
\inst[24]_INST_0\: unisim.vcomponents.LUT6
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
      O => inst(8)
    );
\inst[31]_INST_0\: unisim.vcomponents.LUT6
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
      O => inst(9)
    );
\inst[5]_INST_0\: unisim.vcomponents.LUT6
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
      O => inst(3)
    );
\inst[7]_INST_0\: unisim.vcomponents.LUT6
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
      O => inst(0)
    );
\inst[8]_INST_0\: unisim.vcomponents.LUT6
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
      O => inst(1)
    );
end STRUCTURE;
library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
library UNISIM;
use UNISIM.VCOMPONENTS.ALL;
entity decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix is
  port (
    inst_addr : in STD_LOGIC_VECTOR ( 31 downto 0 );
    inst : out STD_LOGIC_VECTOR ( 31 downto 0 )
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
  signal \^inst\ : STD_LOGIC_VECTOR ( 31 downto 4 );
begin
  inst(31) <= \^inst\(31);
  inst(30) <= \<const0>\;
  inst(29) <= \<const0>\;
  inst(28) <= \<const0>\;
  inst(27) <= \<const0>\;
  inst(26) <= \<const0>\;
  inst(25) <= \<const0>\;
  inst(24 downto 20) <= \^inst\(24 downto 20);
  inst(19) <= \<const0>\;
  inst(18) <= \<const0>\;
  inst(17) <= \<const0>\;
  inst(16) <= \<const0>\;
  inst(15) <= \^inst\(15);
  inst(14) <= \<const0>\;
  inst(13) <= \<const0>\;
  inst(12) <= \<const0>\;
  inst(11 downto 7) <= \^inst\(11 downto 7);
  inst(6) <= \<const0>\;
  inst(5) <= \^inst\(11);
  inst(4) <= \^inst\(4);
  inst(3) <= \<const0>\;
  inst(2) <= \<const0>\;
  inst(1) <= \^inst\(4);
  inst(0) <= \^inst\(4);
GND: unisim.vcomponents.GND
     port map (
      G => \<const0>\
    );
\inst[0]_INST_0\: unisim.vcomponents.LUT5
    generic map(
      INIT => X"0000001F"
    )
        port map (
      I0 => inst_addr(4),
      I1 => inst_addr(3),
      I2 => inst_addr(5),
      I3 => inst_addr(0),
      I4 => inst_addr(1),
      O => \^inst\(4)
    );
\inst[22]_INST_0\: unisim.vcomponents.LUT6
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
      O => \^inst\(22)
    );
\inst[9]_INST_0\: unisim.vcomponents.LUT6
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
      O => \^inst\(9)
    );
\inst__0\: entity work.decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_imem
     port map (
      inst(9) => \^inst\(31),
      inst(8 downto 7) => \^inst\(24 downto 23),
      inst(6 downto 5) => \^inst\(21 downto 20),
      inst(4) => \^inst\(15),
      inst(3 downto 2) => \^inst\(11 downto 10),
      inst(1 downto 0) => \^inst\(8 downto 7),
      inst_addr(5 downto 0) => inst_addr(5 downto 0)
    );
end STRUCTURE;
