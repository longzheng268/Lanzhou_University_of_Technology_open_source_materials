-- Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
-- --------------------------------------------------------------------------------
-- Tool Version: Vivado v.2022.2 (win64) Build 3671981 Fri Oct 14 05:00:03 MDT 2022
-- Date        : Tue Oct 29 12:52:06 2024
-- Host        : cop running 64-bit major release  (build 9200)
-- Command     : write_vhdl -force -mode funcsim
--               e:/project/lab3_risc/lab3_risc.gen/sources_1/bd/risc32/ip/risc32_pc_0_0/risc32_pc_0_0_sim_netlist.vhdl
-- Design      : risc32_pc_0_0
-- Purpose     : This VHDL netlist is a functional simulation representation of the design and should not be modified or
--               synthesized. This netlist cannot be used for SDF annotated simulation.
-- Device      : xc7a100tcsg324-1
-- --------------------------------------------------------------------------------
library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
library UNISIM;
use UNISIM.VCOMPONENTS.ALL;
entity risc32_pc_0_0_pc is
  port (
    q : out STD_LOGIC_VECTOR ( 31 downto 0 );
    data_i : in STD_LOGIC_VECTOR ( 31 downto 0 );
    clk : in STD_LOGIC;
    pc_clr : in STD_LOGIC
  );
  attribute ORIG_REF_NAME : string;
  attribute ORIG_REF_NAME of risc32_pc_0_0_pc : entity is "pc";
end risc32_pc_0_0_pc;

architecture STRUCTURE of risc32_pc_0_0_pc is
  signal \q[31]_i_1_n_0\ : STD_LOGIC;
begin
\q[31]_i_1\: unisim.vcomponents.LUT1
    generic map(
      INIT => X"1"
    )
        port map (
      I0 => pc_clr,
      O => \q[31]_i_1_n_0\
    );
\q_reg[0]\: unisim.vcomponents.FDCE
     port map (
      C => clk,
      CE => '1',
      CLR => \q[31]_i_1_n_0\,
      D => data_i(0),
      Q => q(0)
    );
\q_reg[10]\: unisim.vcomponents.FDCE
     port map (
      C => clk,
      CE => '1',
      CLR => \q[31]_i_1_n_0\,
      D => data_i(10),
      Q => q(10)
    );
\q_reg[11]\: unisim.vcomponents.FDCE
     port map (
      C => clk,
      CE => '1',
      CLR => \q[31]_i_1_n_0\,
      D => data_i(11),
      Q => q(11)
    );
\q_reg[12]\: unisim.vcomponents.FDCE
     port map (
      C => clk,
      CE => '1',
      CLR => \q[31]_i_1_n_0\,
      D => data_i(12),
      Q => q(12)
    );
\q_reg[13]\: unisim.vcomponents.FDCE
     port map (
      C => clk,
      CE => '1',
      CLR => \q[31]_i_1_n_0\,
      D => data_i(13),
      Q => q(13)
    );
\q_reg[14]\: unisim.vcomponents.FDCE
     port map (
      C => clk,
      CE => '1',
      CLR => \q[31]_i_1_n_0\,
      D => data_i(14),
      Q => q(14)
    );
\q_reg[15]\: unisim.vcomponents.FDCE
     port map (
      C => clk,
      CE => '1',
      CLR => \q[31]_i_1_n_0\,
      D => data_i(15),
      Q => q(15)
    );
\q_reg[16]\: unisim.vcomponents.FDCE
     port map (
      C => clk,
      CE => '1',
      CLR => \q[31]_i_1_n_0\,
      D => data_i(16),
      Q => q(16)
    );
\q_reg[17]\: unisim.vcomponents.FDCE
     port map (
      C => clk,
      CE => '1',
      CLR => \q[31]_i_1_n_0\,
      D => data_i(17),
      Q => q(17)
    );
\q_reg[18]\: unisim.vcomponents.FDCE
     port map (
      C => clk,
      CE => '1',
      CLR => \q[31]_i_1_n_0\,
      D => data_i(18),
      Q => q(18)
    );
\q_reg[19]\: unisim.vcomponents.FDCE
     port map (
      C => clk,
      CE => '1',
      CLR => \q[31]_i_1_n_0\,
      D => data_i(19),
      Q => q(19)
    );
\q_reg[1]\: unisim.vcomponents.FDCE
     port map (
      C => clk,
      CE => '1',
      CLR => \q[31]_i_1_n_0\,
      D => data_i(1),
      Q => q(1)
    );
\q_reg[20]\: unisim.vcomponents.FDCE
     port map (
      C => clk,
      CE => '1',
      CLR => \q[31]_i_1_n_0\,
      D => data_i(20),
      Q => q(20)
    );
\q_reg[21]\: unisim.vcomponents.FDCE
     port map (
      C => clk,
      CE => '1',
      CLR => \q[31]_i_1_n_0\,
      D => data_i(21),
      Q => q(21)
    );
\q_reg[22]\: unisim.vcomponents.FDCE
     port map (
      C => clk,
      CE => '1',
      CLR => \q[31]_i_1_n_0\,
      D => data_i(22),
      Q => q(22)
    );
\q_reg[23]\: unisim.vcomponents.FDCE
     port map (
      C => clk,
      CE => '1',
      CLR => \q[31]_i_1_n_0\,
      D => data_i(23),
      Q => q(23)
    );
\q_reg[24]\: unisim.vcomponents.FDCE
     port map (
      C => clk,
      CE => '1',
      CLR => \q[31]_i_1_n_0\,
      D => data_i(24),
      Q => q(24)
    );
\q_reg[25]\: unisim.vcomponents.FDCE
     port map (
      C => clk,
      CE => '1',
      CLR => \q[31]_i_1_n_0\,
      D => data_i(25),
      Q => q(25)
    );
\q_reg[26]\: unisim.vcomponents.FDCE
     port map (
      C => clk,
      CE => '1',
      CLR => \q[31]_i_1_n_0\,
      D => data_i(26),
      Q => q(26)
    );
\q_reg[27]\: unisim.vcomponents.FDCE
     port map (
      C => clk,
      CE => '1',
      CLR => \q[31]_i_1_n_0\,
      D => data_i(27),
      Q => q(27)
    );
\q_reg[28]\: unisim.vcomponents.FDCE
     port map (
      C => clk,
      CE => '1',
      CLR => \q[31]_i_1_n_0\,
      D => data_i(28),
      Q => q(28)
    );
\q_reg[29]\: unisim.vcomponents.FDCE
     port map (
      C => clk,
      CE => '1',
      CLR => \q[31]_i_1_n_0\,
      D => data_i(29),
      Q => q(29)
    );
\q_reg[2]\: unisim.vcomponents.FDCE
     port map (
      C => clk,
      CE => '1',
      CLR => \q[31]_i_1_n_0\,
      D => data_i(2),
      Q => q(2)
    );
\q_reg[30]\: unisim.vcomponents.FDCE
     port map (
      C => clk,
      CE => '1',
      CLR => \q[31]_i_1_n_0\,
      D => data_i(30),
      Q => q(30)
    );
\q_reg[31]\: unisim.vcomponents.FDCE
     port map (
      C => clk,
      CE => '1',
      CLR => \q[31]_i_1_n_0\,
      D => data_i(31),
      Q => q(31)
    );
\q_reg[3]\: unisim.vcomponents.FDCE
     port map (
      C => clk,
      CE => '1',
      CLR => \q[31]_i_1_n_0\,
      D => data_i(3),
      Q => q(3)
    );
\q_reg[4]\: unisim.vcomponents.FDCE
     port map (
      C => clk,
      CE => '1',
      CLR => \q[31]_i_1_n_0\,
      D => data_i(4),
      Q => q(4)
    );
\q_reg[5]\: unisim.vcomponents.FDCE
     port map (
      C => clk,
      CE => '1',
      CLR => \q[31]_i_1_n_0\,
      D => data_i(5),
      Q => q(5)
    );
\q_reg[6]\: unisim.vcomponents.FDCE
     port map (
      C => clk,
      CE => '1',
      CLR => \q[31]_i_1_n_0\,
      D => data_i(6),
      Q => q(6)
    );
\q_reg[7]\: unisim.vcomponents.FDCE
     port map (
      C => clk,
      CE => '1',
      CLR => \q[31]_i_1_n_0\,
      D => data_i(7),
      Q => q(7)
    );
\q_reg[8]\: unisim.vcomponents.FDCE
     port map (
      C => clk,
      CE => '1',
      CLR => \q[31]_i_1_n_0\,
      D => data_i(8),
      Q => q(8)
    );
\q_reg[9]\: unisim.vcomponents.FDCE
     port map (
      C => clk,
      CE => '1',
      CLR => \q[31]_i_1_n_0\,
      D => data_i(9),
      Q => q(9)
    );
end STRUCTURE;
library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
library UNISIM;
use UNISIM.VCOMPONENTS.ALL;
entity risc32_pc_0_0 is
  port (
    clk : in STD_LOGIC;
    pc_clr : in STD_LOGIC;
    data_i : in STD_LOGIC_VECTOR ( 31 downto 0 );
    q : out STD_LOGIC_VECTOR ( 31 downto 0 )
  );
  attribute NotValidForBitStream : boolean;
  attribute NotValidForBitStream of risc32_pc_0_0 : entity is true;
  attribute CHECK_LICENSE_TYPE : string;
  attribute CHECK_LICENSE_TYPE of risc32_pc_0_0 : entity is "risc32_pc_0_0,pc,{}";
  attribute DowngradeIPIdentifiedWarnings : string;
  attribute DowngradeIPIdentifiedWarnings of risc32_pc_0_0 : entity is "yes";
  attribute IP_DEFINITION_SOURCE : string;
  attribute IP_DEFINITION_SOURCE of risc32_pc_0_0 : entity is "module_ref";
  attribute X_CORE_INFO : string;
  attribute X_CORE_INFO of risc32_pc_0_0 : entity is "pc,Vivado 2022.2";
end risc32_pc_0_0;

architecture STRUCTURE of risc32_pc_0_0 is
  attribute X_INTERFACE_INFO : string;
  attribute X_INTERFACE_INFO of clk : signal is "xilinx.com:signal:clock:1.0 clk CLK";
  attribute X_INTERFACE_PARAMETER : string;
  attribute X_INTERFACE_PARAMETER of clk : signal is "XIL_INTERFACENAME clk, FREQ_HZ 100000000, FREQ_TOLERANCE_HZ 0, PHASE 0.0, CLK_DOMAIN risc32_clk, INSERT_VIP 0";
begin
inst: entity work.risc32_pc_0_0_pc
     port map (
      clk => clk,
      data_i(31 downto 0) => data_i(31 downto 0),
      pc_clr => pc_clr,
      q(31 downto 0) => q(31 downto 0)
    );
end STRUCTURE;
