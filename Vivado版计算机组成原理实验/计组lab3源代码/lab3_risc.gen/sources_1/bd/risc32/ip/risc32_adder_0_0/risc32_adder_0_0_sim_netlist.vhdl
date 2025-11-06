-- Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
-- --------------------------------------------------------------------------------
-- Tool Version: Vivado v.2022.2 (win64) Build 3671981 Fri Oct 14 05:00:03 MDT 2022
-- Date        : Mon Oct 28 16:04:53 2024
-- Host        : cop running 64-bit major release  (build 9200)
-- Command     : write_vhdl -force -mode funcsim
--               e:/project/lab3_risc/lab3_risc.gen/sources_1/bd/risc32/ip/risc32_adder_0_0/risc32_adder_0_0_sim_netlist.vhdl
-- Design      : risc32_adder_0_0
-- Purpose     : This VHDL netlist is a functional simulation representation of the design and should not be modified or
--               synthesized. This netlist cannot be used for SDF annotated simulation.
-- Device      : xc7a100tcsg324-1
-- --------------------------------------------------------------------------------
library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
library UNISIM;
use UNISIM.VCOMPONENTS.ALL;
entity risc32_adder_0_0_adder is
  port (
    add_out : out STD_LOGIC_VECTOR ( 31 downto 0 );
    add_a : in STD_LOGIC_VECTOR ( 31 downto 0 );
    add_b : in STD_LOGIC_VECTOR ( 31 downto 0 )
  );
  attribute ORIG_REF_NAME : string;
  attribute ORIG_REF_NAME of risc32_adder_0_0_adder : entity is "adder";
end risc32_adder_0_0_adder;

architecture STRUCTURE of risc32_adder_0_0_adder is
  signal \add_out[0]_INST_0_i_1_n_0\ : STD_LOGIC;
  signal \add_out[0]_INST_0_i_2_n_0\ : STD_LOGIC;
  signal \add_out[0]_INST_0_i_3_n_0\ : STD_LOGIC;
  signal \add_out[0]_INST_0_i_4_n_0\ : STD_LOGIC;
  signal \add_out[0]_INST_0_n_0\ : STD_LOGIC;
  signal \add_out[0]_INST_0_n_1\ : STD_LOGIC;
  signal \add_out[0]_INST_0_n_2\ : STD_LOGIC;
  signal \add_out[0]_INST_0_n_3\ : STD_LOGIC;
  signal \add_out[12]_INST_0_i_1_n_0\ : STD_LOGIC;
  signal \add_out[12]_INST_0_i_2_n_0\ : STD_LOGIC;
  signal \add_out[12]_INST_0_i_3_n_0\ : STD_LOGIC;
  signal \add_out[12]_INST_0_i_4_n_0\ : STD_LOGIC;
  signal \add_out[12]_INST_0_n_0\ : STD_LOGIC;
  signal \add_out[12]_INST_0_n_1\ : STD_LOGIC;
  signal \add_out[12]_INST_0_n_2\ : STD_LOGIC;
  signal \add_out[12]_INST_0_n_3\ : STD_LOGIC;
  signal \add_out[16]_INST_0_i_1_n_0\ : STD_LOGIC;
  signal \add_out[16]_INST_0_i_2_n_0\ : STD_LOGIC;
  signal \add_out[16]_INST_0_i_3_n_0\ : STD_LOGIC;
  signal \add_out[16]_INST_0_i_4_n_0\ : STD_LOGIC;
  signal \add_out[16]_INST_0_n_0\ : STD_LOGIC;
  signal \add_out[16]_INST_0_n_1\ : STD_LOGIC;
  signal \add_out[16]_INST_0_n_2\ : STD_LOGIC;
  signal \add_out[16]_INST_0_n_3\ : STD_LOGIC;
  signal \add_out[20]_INST_0_i_1_n_0\ : STD_LOGIC;
  signal \add_out[20]_INST_0_i_2_n_0\ : STD_LOGIC;
  signal \add_out[20]_INST_0_i_3_n_0\ : STD_LOGIC;
  signal \add_out[20]_INST_0_i_4_n_0\ : STD_LOGIC;
  signal \add_out[20]_INST_0_n_0\ : STD_LOGIC;
  signal \add_out[20]_INST_0_n_1\ : STD_LOGIC;
  signal \add_out[20]_INST_0_n_2\ : STD_LOGIC;
  signal \add_out[20]_INST_0_n_3\ : STD_LOGIC;
  signal \add_out[24]_INST_0_i_1_n_0\ : STD_LOGIC;
  signal \add_out[24]_INST_0_i_2_n_0\ : STD_LOGIC;
  signal \add_out[24]_INST_0_i_3_n_0\ : STD_LOGIC;
  signal \add_out[24]_INST_0_i_4_n_0\ : STD_LOGIC;
  signal \add_out[24]_INST_0_n_0\ : STD_LOGIC;
  signal \add_out[24]_INST_0_n_1\ : STD_LOGIC;
  signal \add_out[24]_INST_0_n_2\ : STD_LOGIC;
  signal \add_out[24]_INST_0_n_3\ : STD_LOGIC;
  signal \add_out[28]_INST_0_i_1_n_0\ : STD_LOGIC;
  signal \add_out[28]_INST_0_i_2_n_0\ : STD_LOGIC;
  signal \add_out[28]_INST_0_i_3_n_0\ : STD_LOGIC;
  signal \add_out[28]_INST_0_i_4_n_0\ : STD_LOGIC;
  signal \add_out[28]_INST_0_n_1\ : STD_LOGIC;
  signal \add_out[28]_INST_0_n_2\ : STD_LOGIC;
  signal \add_out[28]_INST_0_n_3\ : STD_LOGIC;
  signal \add_out[4]_INST_0_i_1_n_0\ : STD_LOGIC;
  signal \add_out[4]_INST_0_i_2_n_0\ : STD_LOGIC;
  signal \add_out[4]_INST_0_i_3_n_0\ : STD_LOGIC;
  signal \add_out[4]_INST_0_i_4_n_0\ : STD_LOGIC;
  signal \add_out[4]_INST_0_n_0\ : STD_LOGIC;
  signal \add_out[4]_INST_0_n_1\ : STD_LOGIC;
  signal \add_out[4]_INST_0_n_2\ : STD_LOGIC;
  signal \add_out[4]_INST_0_n_3\ : STD_LOGIC;
  signal \add_out[8]_INST_0_i_1_n_0\ : STD_LOGIC;
  signal \add_out[8]_INST_0_i_2_n_0\ : STD_LOGIC;
  signal \add_out[8]_INST_0_i_3_n_0\ : STD_LOGIC;
  signal \add_out[8]_INST_0_i_4_n_0\ : STD_LOGIC;
  signal \add_out[8]_INST_0_n_0\ : STD_LOGIC;
  signal \add_out[8]_INST_0_n_1\ : STD_LOGIC;
  signal \add_out[8]_INST_0_n_2\ : STD_LOGIC;
  signal \add_out[8]_INST_0_n_3\ : STD_LOGIC;
  signal \NLW_add_out[28]_INST_0_CO_UNCONNECTED\ : STD_LOGIC_VECTOR ( 3 to 3 );
  attribute ADDER_THRESHOLD : integer;
  attribute ADDER_THRESHOLD of \add_out[0]_INST_0\ : label is 35;
  attribute ADDER_THRESHOLD of \add_out[12]_INST_0\ : label is 35;
  attribute ADDER_THRESHOLD of \add_out[16]_INST_0\ : label is 35;
  attribute ADDER_THRESHOLD of \add_out[20]_INST_0\ : label is 35;
  attribute ADDER_THRESHOLD of \add_out[24]_INST_0\ : label is 35;
  attribute ADDER_THRESHOLD of \add_out[28]_INST_0\ : label is 35;
  attribute ADDER_THRESHOLD of \add_out[4]_INST_0\ : label is 35;
  attribute ADDER_THRESHOLD of \add_out[8]_INST_0\ : label is 35;
begin
\add_out[0]_INST_0\: unisim.vcomponents.CARRY4
     port map (
      CI => '0',
      CO(3) => \add_out[0]_INST_0_n_0\,
      CO(2) => \add_out[0]_INST_0_n_1\,
      CO(1) => \add_out[0]_INST_0_n_2\,
      CO(0) => \add_out[0]_INST_0_n_3\,
      CYINIT => '0',
      DI(3 downto 0) => add_a(3 downto 0),
      O(3 downto 0) => add_out(3 downto 0),
      S(3) => \add_out[0]_INST_0_i_1_n_0\,
      S(2) => \add_out[0]_INST_0_i_2_n_0\,
      S(1) => \add_out[0]_INST_0_i_3_n_0\,
      S(0) => \add_out[0]_INST_0_i_4_n_0\
    );
\add_out[0]_INST_0_i_1\: unisim.vcomponents.LUT2
    generic map(
      INIT => X"6"
    )
        port map (
      I0 => add_a(3),
      I1 => add_b(3),
      O => \add_out[0]_INST_0_i_1_n_0\
    );
\add_out[0]_INST_0_i_2\: unisim.vcomponents.LUT2
    generic map(
      INIT => X"6"
    )
        port map (
      I0 => add_a(2),
      I1 => add_b(2),
      O => \add_out[0]_INST_0_i_2_n_0\
    );
\add_out[0]_INST_0_i_3\: unisim.vcomponents.LUT2
    generic map(
      INIT => X"6"
    )
        port map (
      I0 => add_a(1),
      I1 => add_b(1),
      O => \add_out[0]_INST_0_i_3_n_0\
    );
\add_out[0]_INST_0_i_4\: unisim.vcomponents.LUT2
    generic map(
      INIT => X"6"
    )
        port map (
      I0 => add_a(0),
      I1 => add_b(0),
      O => \add_out[0]_INST_0_i_4_n_0\
    );
\add_out[12]_INST_0\: unisim.vcomponents.CARRY4
     port map (
      CI => \add_out[8]_INST_0_n_0\,
      CO(3) => \add_out[12]_INST_0_n_0\,
      CO(2) => \add_out[12]_INST_0_n_1\,
      CO(1) => \add_out[12]_INST_0_n_2\,
      CO(0) => \add_out[12]_INST_0_n_3\,
      CYINIT => '0',
      DI(3 downto 0) => add_a(15 downto 12),
      O(3 downto 0) => add_out(15 downto 12),
      S(3) => \add_out[12]_INST_0_i_1_n_0\,
      S(2) => \add_out[12]_INST_0_i_2_n_0\,
      S(1) => \add_out[12]_INST_0_i_3_n_0\,
      S(0) => \add_out[12]_INST_0_i_4_n_0\
    );
\add_out[12]_INST_0_i_1\: unisim.vcomponents.LUT2
    generic map(
      INIT => X"6"
    )
        port map (
      I0 => add_a(15),
      I1 => add_b(15),
      O => \add_out[12]_INST_0_i_1_n_0\
    );
\add_out[12]_INST_0_i_2\: unisim.vcomponents.LUT2
    generic map(
      INIT => X"6"
    )
        port map (
      I0 => add_a(14),
      I1 => add_b(14),
      O => \add_out[12]_INST_0_i_2_n_0\
    );
\add_out[12]_INST_0_i_3\: unisim.vcomponents.LUT2
    generic map(
      INIT => X"6"
    )
        port map (
      I0 => add_a(13),
      I1 => add_b(13),
      O => \add_out[12]_INST_0_i_3_n_0\
    );
\add_out[12]_INST_0_i_4\: unisim.vcomponents.LUT2
    generic map(
      INIT => X"6"
    )
        port map (
      I0 => add_a(12),
      I1 => add_b(12),
      O => \add_out[12]_INST_0_i_4_n_0\
    );
\add_out[16]_INST_0\: unisim.vcomponents.CARRY4
     port map (
      CI => \add_out[12]_INST_0_n_0\,
      CO(3) => \add_out[16]_INST_0_n_0\,
      CO(2) => \add_out[16]_INST_0_n_1\,
      CO(1) => \add_out[16]_INST_0_n_2\,
      CO(0) => \add_out[16]_INST_0_n_3\,
      CYINIT => '0',
      DI(3 downto 0) => add_a(19 downto 16),
      O(3 downto 0) => add_out(19 downto 16),
      S(3) => \add_out[16]_INST_0_i_1_n_0\,
      S(2) => \add_out[16]_INST_0_i_2_n_0\,
      S(1) => \add_out[16]_INST_0_i_3_n_0\,
      S(0) => \add_out[16]_INST_0_i_4_n_0\
    );
\add_out[16]_INST_0_i_1\: unisim.vcomponents.LUT2
    generic map(
      INIT => X"6"
    )
        port map (
      I0 => add_a(19),
      I1 => add_b(19),
      O => \add_out[16]_INST_0_i_1_n_0\
    );
\add_out[16]_INST_0_i_2\: unisim.vcomponents.LUT2
    generic map(
      INIT => X"6"
    )
        port map (
      I0 => add_a(18),
      I1 => add_b(18),
      O => \add_out[16]_INST_0_i_2_n_0\
    );
\add_out[16]_INST_0_i_3\: unisim.vcomponents.LUT2
    generic map(
      INIT => X"6"
    )
        port map (
      I0 => add_a(17),
      I1 => add_b(17),
      O => \add_out[16]_INST_0_i_3_n_0\
    );
\add_out[16]_INST_0_i_4\: unisim.vcomponents.LUT2
    generic map(
      INIT => X"6"
    )
        port map (
      I0 => add_a(16),
      I1 => add_b(16),
      O => \add_out[16]_INST_0_i_4_n_0\
    );
\add_out[20]_INST_0\: unisim.vcomponents.CARRY4
     port map (
      CI => \add_out[16]_INST_0_n_0\,
      CO(3) => \add_out[20]_INST_0_n_0\,
      CO(2) => \add_out[20]_INST_0_n_1\,
      CO(1) => \add_out[20]_INST_0_n_2\,
      CO(0) => \add_out[20]_INST_0_n_3\,
      CYINIT => '0',
      DI(3 downto 0) => add_a(23 downto 20),
      O(3 downto 0) => add_out(23 downto 20),
      S(3) => \add_out[20]_INST_0_i_1_n_0\,
      S(2) => \add_out[20]_INST_0_i_2_n_0\,
      S(1) => \add_out[20]_INST_0_i_3_n_0\,
      S(0) => \add_out[20]_INST_0_i_4_n_0\
    );
\add_out[20]_INST_0_i_1\: unisim.vcomponents.LUT2
    generic map(
      INIT => X"6"
    )
        port map (
      I0 => add_a(23),
      I1 => add_b(23),
      O => \add_out[20]_INST_0_i_1_n_0\
    );
\add_out[20]_INST_0_i_2\: unisim.vcomponents.LUT2
    generic map(
      INIT => X"6"
    )
        port map (
      I0 => add_a(22),
      I1 => add_b(22),
      O => \add_out[20]_INST_0_i_2_n_0\
    );
\add_out[20]_INST_0_i_3\: unisim.vcomponents.LUT2
    generic map(
      INIT => X"6"
    )
        port map (
      I0 => add_a(21),
      I1 => add_b(21),
      O => \add_out[20]_INST_0_i_3_n_0\
    );
\add_out[20]_INST_0_i_4\: unisim.vcomponents.LUT2
    generic map(
      INIT => X"6"
    )
        port map (
      I0 => add_a(20),
      I1 => add_b(20),
      O => \add_out[20]_INST_0_i_4_n_0\
    );
\add_out[24]_INST_0\: unisim.vcomponents.CARRY4
     port map (
      CI => \add_out[20]_INST_0_n_0\,
      CO(3) => \add_out[24]_INST_0_n_0\,
      CO(2) => \add_out[24]_INST_0_n_1\,
      CO(1) => \add_out[24]_INST_0_n_2\,
      CO(0) => \add_out[24]_INST_0_n_3\,
      CYINIT => '0',
      DI(3 downto 0) => add_a(27 downto 24),
      O(3 downto 0) => add_out(27 downto 24),
      S(3) => \add_out[24]_INST_0_i_1_n_0\,
      S(2) => \add_out[24]_INST_0_i_2_n_0\,
      S(1) => \add_out[24]_INST_0_i_3_n_0\,
      S(0) => \add_out[24]_INST_0_i_4_n_0\
    );
\add_out[24]_INST_0_i_1\: unisim.vcomponents.LUT2
    generic map(
      INIT => X"6"
    )
        port map (
      I0 => add_a(27),
      I1 => add_b(27),
      O => \add_out[24]_INST_0_i_1_n_0\
    );
\add_out[24]_INST_0_i_2\: unisim.vcomponents.LUT2
    generic map(
      INIT => X"6"
    )
        port map (
      I0 => add_a(26),
      I1 => add_b(26),
      O => \add_out[24]_INST_0_i_2_n_0\
    );
\add_out[24]_INST_0_i_3\: unisim.vcomponents.LUT2
    generic map(
      INIT => X"6"
    )
        port map (
      I0 => add_a(25),
      I1 => add_b(25),
      O => \add_out[24]_INST_0_i_3_n_0\
    );
\add_out[24]_INST_0_i_4\: unisim.vcomponents.LUT2
    generic map(
      INIT => X"6"
    )
        port map (
      I0 => add_a(24),
      I1 => add_b(24),
      O => \add_out[24]_INST_0_i_4_n_0\
    );
\add_out[28]_INST_0\: unisim.vcomponents.CARRY4
     port map (
      CI => \add_out[24]_INST_0_n_0\,
      CO(3) => \NLW_add_out[28]_INST_0_CO_UNCONNECTED\(3),
      CO(2) => \add_out[28]_INST_0_n_1\,
      CO(1) => \add_out[28]_INST_0_n_2\,
      CO(0) => \add_out[28]_INST_0_n_3\,
      CYINIT => '0',
      DI(3) => '0',
      DI(2 downto 0) => add_a(30 downto 28),
      O(3 downto 0) => add_out(31 downto 28),
      S(3) => \add_out[28]_INST_0_i_1_n_0\,
      S(2) => \add_out[28]_INST_0_i_2_n_0\,
      S(1) => \add_out[28]_INST_0_i_3_n_0\,
      S(0) => \add_out[28]_INST_0_i_4_n_0\
    );
\add_out[28]_INST_0_i_1\: unisim.vcomponents.LUT2
    generic map(
      INIT => X"6"
    )
        port map (
      I0 => add_a(31),
      I1 => add_b(31),
      O => \add_out[28]_INST_0_i_1_n_0\
    );
\add_out[28]_INST_0_i_2\: unisim.vcomponents.LUT2
    generic map(
      INIT => X"6"
    )
        port map (
      I0 => add_a(30),
      I1 => add_b(30),
      O => \add_out[28]_INST_0_i_2_n_0\
    );
\add_out[28]_INST_0_i_3\: unisim.vcomponents.LUT2
    generic map(
      INIT => X"6"
    )
        port map (
      I0 => add_a(29),
      I1 => add_b(29),
      O => \add_out[28]_INST_0_i_3_n_0\
    );
\add_out[28]_INST_0_i_4\: unisim.vcomponents.LUT2
    generic map(
      INIT => X"6"
    )
        port map (
      I0 => add_a(28),
      I1 => add_b(28),
      O => \add_out[28]_INST_0_i_4_n_0\
    );
\add_out[4]_INST_0\: unisim.vcomponents.CARRY4
     port map (
      CI => \add_out[0]_INST_0_n_0\,
      CO(3) => \add_out[4]_INST_0_n_0\,
      CO(2) => \add_out[4]_INST_0_n_1\,
      CO(1) => \add_out[4]_INST_0_n_2\,
      CO(0) => \add_out[4]_INST_0_n_3\,
      CYINIT => '0',
      DI(3 downto 0) => add_a(7 downto 4),
      O(3 downto 0) => add_out(7 downto 4),
      S(3) => \add_out[4]_INST_0_i_1_n_0\,
      S(2) => \add_out[4]_INST_0_i_2_n_0\,
      S(1) => \add_out[4]_INST_0_i_3_n_0\,
      S(0) => \add_out[4]_INST_0_i_4_n_0\
    );
\add_out[4]_INST_0_i_1\: unisim.vcomponents.LUT2
    generic map(
      INIT => X"6"
    )
        port map (
      I0 => add_a(7),
      I1 => add_b(7),
      O => \add_out[4]_INST_0_i_1_n_0\
    );
\add_out[4]_INST_0_i_2\: unisim.vcomponents.LUT2
    generic map(
      INIT => X"6"
    )
        port map (
      I0 => add_a(6),
      I1 => add_b(6),
      O => \add_out[4]_INST_0_i_2_n_0\
    );
\add_out[4]_INST_0_i_3\: unisim.vcomponents.LUT2
    generic map(
      INIT => X"6"
    )
        port map (
      I0 => add_a(5),
      I1 => add_b(5),
      O => \add_out[4]_INST_0_i_3_n_0\
    );
\add_out[4]_INST_0_i_4\: unisim.vcomponents.LUT2
    generic map(
      INIT => X"6"
    )
        port map (
      I0 => add_a(4),
      I1 => add_b(4),
      O => \add_out[4]_INST_0_i_4_n_0\
    );
\add_out[8]_INST_0\: unisim.vcomponents.CARRY4
     port map (
      CI => \add_out[4]_INST_0_n_0\,
      CO(3) => \add_out[8]_INST_0_n_0\,
      CO(2) => \add_out[8]_INST_0_n_1\,
      CO(1) => \add_out[8]_INST_0_n_2\,
      CO(0) => \add_out[8]_INST_0_n_3\,
      CYINIT => '0',
      DI(3 downto 0) => add_a(11 downto 8),
      O(3 downto 0) => add_out(11 downto 8),
      S(3) => \add_out[8]_INST_0_i_1_n_0\,
      S(2) => \add_out[8]_INST_0_i_2_n_0\,
      S(1) => \add_out[8]_INST_0_i_3_n_0\,
      S(0) => \add_out[8]_INST_0_i_4_n_0\
    );
\add_out[8]_INST_0_i_1\: unisim.vcomponents.LUT2
    generic map(
      INIT => X"6"
    )
        port map (
      I0 => add_a(11),
      I1 => add_b(11),
      O => \add_out[8]_INST_0_i_1_n_0\
    );
\add_out[8]_INST_0_i_2\: unisim.vcomponents.LUT2
    generic map(
      INIT => X"6"
    )
        port map (
      I0 => add_a(10),
      I1 => add_b(10),
      O => \add_out[8]_INST_0_i_2_n_0\
    );
\add_out[8]_INST_0_i_3\: unisim.vcomponents.LUT2
    generic map(
      INIT => X"6"
    )
        port map (
      I0 => add_a(9),
      I1 => add_b(9),
      O => \add_out[8]_INST_0_i_3_n_0\
    );
\add_out[8]_INST_0_i_4\: unisim.vcomponents.LUT2
    generic map(
      INIT => X"6"
    )
        port map (
      I0 => add_a(8),
      I1 => add_b(8),
      O => \add_out[8]_INST_0_i_4_n_0\
    );
end STRUCTURE;
library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
library UNISIM;
use UNISIM.VCOMPONENTS.ALL;
entity risc32_adder_0_0 is
  port (
    add_a : in STD_LOGIC_VECTOR ( 31 downto 0 );
    add_b : in STD_LOGIC_VECTOR ( 31 downto 0 );
    add_out : out STD_LOGIC_VECTOR ( 31 downto 0 )
  );
  attribute NotValidForBitStream : boolean;
  attribute NotValidForBitStream of risc32_adder_0_0 : entity is true;
  attribute CHECK_LICENSE_TYPE : string;
  attribute CHECK_LICENSE_TYPE of risc32_adder_0_0 : entity is "risc32_adder_0_0,adder,{}";
  attribute DowngradeIPIdentifiedWarnings : string;
  attribute DowngradeIPIdentifiedWarnings of risc32_adder_0_0 : entity is "yes";
  attribute IP_DEFINITION_SOURCE : string;
  attribute IP_DEFINITION_SOURCE of risc32_adder_0_0 : entity is "module_ref";
  attribute X_CORE_INFO : string;
  attribute X_CORE_INFO of risc32_adder_0_0 : entity is "adder,Vivado 2022.2";
end risc32_adder_0_0;

architecture STRUCTURE of risc32_adder_0_0 is
begin
inst: entity work.risc32_adder_0_0_adder
     port map (
      add_a(31 downto 0) => add_a(31 downto 0),
      add_b(31 downto 0) => add_b(31 downto 0),
      add_out(31 downto 0) => add_out(31 downto 0)
    );
end STRUCTURE;
