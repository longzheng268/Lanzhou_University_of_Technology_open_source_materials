-- Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
-- --------------------------------------------------------------------------------
-- Tool Version: Vivado v.2022.2 (win64) Build 3671981 Fri Oct 14 05:00:03 MDT 2022
-- Date        : Tue Oct 29 10:56:54 2024
-- Host        : cop running 64-bit major release  (build 9200)
-- Command     : write_vhdl -force -mode funcsim
--               e:/project/lab3_risc/lab3_risc.gen/sources_1/bd/risc32/ip/risc32_segment_0_0/risc32_segment_0_0_sim_netlist.vhdl
-- Design      : risc32_segment_0_0
-- Purpose     : This VHDL netlist is a functional simulation representation of the design and should not be modified or
--               synthesized. This netlist cannot be used for SDF annotated simulation.
-- Device      : xc7a100tcsg324-1
-- --------------------------------------------------------------------------------
library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
library UNISIM;
use UNISIM.VCOMPONENTS.ALL;
entity risc32_segment_0_0_segment is
  port (
    AN : out STD_LOGIC_VECTOR ( 7 downto 0 );
    seg_data_o : out STD_LOGIC_VECTOR ( 6 downto 0 );
    clk : in STD_LOGIC;
    Data_i : in STD_LOGIC_VECTOR ( 31 downto 0 );
    rst_n : in STD_LOGIC
  );
  attribute ORIG_REF_NAME : string;
  attribute ORIG_REF_NAME of risc32_segment_0_0_segment : entity is "segment";
end risc32_segment_0_0_segment;

architecture STRUCTURE of risc32_segment_0_0_segment is
  signal \AN[0]_i_1_n_0\ : STD_LOGIC;
  signal \AN[1]_i_1_n_0\ : STD_LOGIC;
  signal \AN[2]_i_1_n_0\ : STD_LOGIC;
  signal \AN[3]_i_1_n_0\ : STD_LOGIC;
  signal \AN[4]_i_1_n_0\ : STD_LOGIC;
  signal \AN[5]_i_1_n_0\ : STD_LOGIC;
  signal \AN[6]_i_1_n_0\ : STD_LOGIC;
  signal \AN[7]_i_1_n_0\ : STD_LOGIC;
  signal \AN[7]_i_2_n_0\ : STD_LOGIC;
  signal cnt : STD_LOGIC_VECTOR ( 2 downto 0 );
  signal \cnt[0]_i_1_n_0\ : STD_LOGIC;
  signal \cnt[1]_i_1_n_0\ : STD_LOGIC;
  signal \cnt[2]_i_1_n_0\ : STD_LOGIC;
  signal cnt_ms : STD_LOGIC_VECTOR ( 15 downto 0 );
  signal \cnt_ms0_carry__0_n_0\ : STD_LOGIC;
  signal \cnt_ms0_carry__0_n_1\ : STD_LOGIC;
  signal \cnt_ms0_carry__0_n_2\ : STD_LOGIC;
  signal \cnt_ms0_carry__0_n_3\ : STD_LOGIC;
  signal \cnt_ms0_carry__1_n_0\ : STD_LOGIC;
  signal \cnt_ms0_carry__1_n_1\ : STD_LOGIC;
  signal \cnt_ms0_carry__1_n_2\ : STD_LOGIC;
  signal \cnt_ms0_carry__1_n_3\ : STD_LOGIC;
  signal \cnt_ms0_carry__2_n_2\ : STD_LOGIC;
  signal \cnt_ms0_carry__2_n_3\ : STD_LOGIC;
  signal cnt_ms0_carry_n_0 : STD_LOGIC;
  signal cnt_ms0_carry_n_1 : STD_LOGIC;
  signal cnt_ms0_carry_n_2 : STD_LOGIC;
  signal cnt_ms0_carry_n_3 : STD_LOGIC;
  signal \cnt_ms[15]_i_2_n_0\ : STD_LOGIC;
  signal \cnt_ms[15]_i_3_n_0\ : STD_LOGIC;
  signal \cnt_ms[15]_i_4_n_0\ : STD_LOGIC;
  signal \cnt_ms[15]_i_5_n_0\ : STD_LOGIC;
  signal \cnt_ms[15]_i_6_n_0\ : STD_LOGIC;
  signal cnt_ms_1 : STD_LOGIC_VECTOR ( 15 downto 0 );
  signal data0 : STD_LOGIC_VECTOR ( 15 downto 1 );
  signal seg_data_o_0 : STD_LOGIC_VECTOR ( 6 downto 0 );
  signal seg_num : STD_LOGIC_VECTOR ( 3 downto 0 );
  signal \seg_num[0]_i_2_n_0\ : STD_LOGIC;
  signal \seg_num[0]_i_3_n_0\ : STD_LOGIC;
  signal \seg_num[1]_i_2_n_0\ : STD_LOGIC;
  signal \seg_num[1]_i_3_n_0\ : STD_LOGIC;
  signal \seg_num[2]_i_2_n_0\ : STD_LOGIC;
  signal \seg_num[2]_i_3_n_0\ : STD_LOGIC;
  signal \seg_num[3]_i_2_n_0\ : STD_LOGIC;
  signal \seg_num[3]_i_3_n_0\ : STD_LOGIC;
  signal seg_num_2 : STD_LOGIC_VECTOR ( 3 downto 0 );
  signal \NLW_cnt_ms0_carry__2_CO_UNCONNECTED\ : STD_LOGIC_VECTOR ( 3 downto 2 );
  signal \NLW_cnt_ms0_carry__2_O_UNCONNECTED\ : STD_LOGIC_VECTOR ( 3 to 3 );
  attribute SOFT_HLUTNM : string;
  attribute SOFT_HLUTNM of \AN[0]_i_1\ : label is "soft_lutpair5";
  attribute SOFT_HLUTNM of \AN[1]_i_1\ : label is "soft_lutpair5";
  attribute SOFT_HLUTNM of \AN[2]_i_1\ : label is "soft_lutpair6";
  attribute SOFT_HLUTNM of \AN[3]_i_1\ : label is "soft_lutpair6";
  attribute SOFT_HLUTNM of \AN[4]_i_1\ : label is "soft_lutpair7";
  attribute SOFT_HLUTNM of \AN[5]_i_1\ : label is "soft_lutpair7";
  attribute SOFT_HLUTNM of \AN[6]_i_1\ : label is "soft_lutpair8";
  attribute SOFT_HLUTNM of \AN[7]_i_1\ : label is "soft_lutpair8";
  attribute SOFT_HLUTNM of \cnt[0]_i_1\ : label is "soft_lutpair9";
  attribute SOFT_HLUTNM of \cnt[1]_i_1\ : label is "soft_lutpair0";
  attribute SOFT_HLUTNM of \cnt[2]_i_1\ : label is "soft_lutpair0";
  attribute ADDER_THRESHOLD : integer;
  attribute ADDER_THRESHOLD of cnt_ms0_carry : label is 35;
  attribute ADDER_THRESHOLD of \cnt_ms0_carry__0\ : label is 35;
  attribute ADDER_THRESHOLD of \cnt_ms0_carry__1\ : label is 35;
  attribute ADDER_THRESHOLD of \cnt_ms0_carry__2\ : label is 35;
  attribute SOFT_HLUTNM of \cnt_ms[0]_i_1\ : label is "soft_lutpair4";
  attribute SOFT_HLUTNM of \cnt_ms[10]_i_1\ : label is "soft_lutpair14";
  attribute SOFT_HLUTNM of \cnt_ms[11]_i_1\ : label is "soft_lutpair14";
  attribute SOFT_HLUTNM of \cnt_ms[12]_i_1\ : label is "soft_lutpair15";
  attribute SOFT_HLUTNM of \cnt_ms[13]_i_1\ : label is "soft_lutpair15";
  attribute SOFT_HLUTNM of \cnt_ms[14]_i_1\ : label is "soft_lutpair16";
  attribute SOFT_HLUTNM of \cnt_ms[15]_i_1\ : label is "soft_lutpair16";
  attribute SOFT_HLUTNM of \cnt_ms[15]_i_4\ : label is "soft_lutpair4";
  attribute SOFT_HLUTNM of \cnt_ms[1]_i_1\ : label is "soft_lutpair9";
  attribute SOFT_HLUTNM of \cnt_ms[2]_i_1\ : label is "soft_lutpair10";
  attribute SOFT_HLUTNM of \cnt_ms[3]_i_1\ : label is "soft_lutpair10";
  attribute SOFT_HLUTNM of \cnt_ms[4]_i_1\ : label is "soft_lutpair11";
  attribute SOFT_HLUTNM of \cnt_ms[5]_i_1\ : label is "soft_lutpair11";
  attribute SOFT_HLUTNM of \cnt_ms[6]_i_1\ : label is "soft_lutpair12";
  attribute SOFT_HLUTNM of \cnt_ms[7]_i_1\ : label is "soft_lutpair12";
  attribute SOFT_HLUTNM of \cnt_ms[8]_i_1\ : label is "soft_lutpair13";
  attribute SOFT_HLUTNM of \cnt_ms[9]_i_1\ : label is "soft_lutpair13";
  attribute SOFT_HLUTNM of \seg_data_o[0]_i_1\ : label is "soft_lutpair1";
  attribute SOFT_HLUTNM of \seg_data_o[1]_i_1\ : label is "soft_lutpair1";
  attribute SOFT_HLUTNM of \seg_data_o[2]_i_1\ : label is "soft_lutpair2";
  attribute SOFT_HLUTNM of \seg_data_o[3]_i_1\ : label is "soft_lutpair2";
  attribute SOFT_HLUTNM of \seg_data_o[4]_i_1\ : label is "soft_lutpair3";
  attribute SOFT_HLUTNM of \seg_data_o[5]_i_1\ : label is "soft_lutpair3";
begin
\AN[0]_i_1\: unisim.vcomponents.LUT3
    generic map(
      INIT => X"FE"
    )
        port map (
      I0 => cnt(1),
      I1 => cnt(2),
      I2 => cnt(0),
      O => \AN[0]_i_1_n_0\
    );
\AN[1]_i_1\: unisim.vcomponents.LUT3
    generic map(
      INIT => X"EF"
    )
        port map (
      I0 => cnt(1),
      I1 => cnt(2),
      I2 => cnt(0),
      O => \AN[1]_i_1_n_0\
    );
\AN[2]_i_1\: unisim.vcomponents.LUT3
    generic map(
      INIT => X"EF"
    )
        port map (
      I0 => cnt(2),
      I1 => cnt(0),
      I2 => cnt(1),
      O => \AN[2]_i_1_n_0\
    );
\AN[3]_i_1\: unisim.vcomponents.LUT3
    generic map(
      INIT => X"F7"
    )
        port map (
      I0 => cnt(1),
      I1 => cnt(0),
      I2 => cnt(2),
      O => \AN[3]_i_1_n_0\
    );
\AN[4]_i_1\: unisim.vcomponents.LUT3
    generic map(
      INIT => X"EF"
    )
        port map (
      I0 => cnt(1),
      I1 => cnt(0),
      I2 => cnt(2),
      O => \AN[4]_i_1_n_0\
    );
\AN[5]_i_1\: unisim.vcomponents.LUT3
    generic map(
      INIT => X"F7"
    )
        port map (
      I0 => cnt(2),
      I1 => cnt(0),
      I2 => cnt(1),
      O => \AN[5]_i_1_n_0\
    );
\AN[6]_i_1\: unisim.vcomponents.LUT3
    generic map(
      INIT => X"F7"
    )
        port map (
      I0 => cnt(1),
      I1 => cnt(2),
      I2 => cnt(0),
      O => \AN[6]_i_1_n_0\
    );
\AN[7]_i_1\: unisim.vcomponents.LUT3
    generic map(
      INIT => X"7F"
    )
        port map (
      I0 => cnt(2),
      I1 => cnt(0),
      I2 => cnt(1),
      O => \AN[7]_i_1_n_0\
    );
\AN[7]_i_2\: unisim.vcomponents.LUT1
    generic map(
      INIT => X"1"
    )
        port map (
      I0 => rst_n,
      O => \AN[7]_i_2_n_0\
    );
\AN_reg[0]\: unisim.vcomponents.FDPE
     port map (
      C => clk,
      CE => '1',
      D => \AN[0]_i_1_n_0\,
      PRE => \AN[7]_i_2_n_0\,
      Q => AN(0)
    );
\AN_reg[1]\: unisim.vcomponents.FDPE
     port map (
      C => clk,
      CE => '1',
      D => \AN[1]_i_1_n_0\,
      PRE => \AN[7]_i_2_n_0\,
      Q => AN(1)
    );
\AN_reg[2]\: unisim.vcomponents.FDPE
     port map (
      C => clk,
      CE => '1',
      D => \AN[2]_i_1_n_0\,
      PRE => \AN[7]_i_2_n_0\,
      Q => AN(2)
    );
\AN_reg[3]\: unisim.vcomponents.FDPE
     port map (
      C => clk,
      CE => '1',
      D => \AN[3]_i_1_n_0\,
      PRE => \AN[7]_i_2_n_0\,
      Q => AN(3)
    );
\AN_reg[4]\: unisim.vcomponents.FDPE
     port map (
      C => clk,
      CE => '1',
      D => \AN[4]_i_1_n_0\,
      PRE => \AN[7]_i_2_n_0\,
      Q => AN(4)
    );
\AN_reg[5]\: unisim.vcomponents.FDPE
     port map (
      C => clk,
      CE => '1',
      D => \AN[5]_i_1_n_0\,
      PRE => \AN[7]_i_2_n_0\,
      Q => AN(5)
    );
\AN_reg[6]\: unisim.vcomponents.FDPE
     port map (
      C => clk,
      CE => '1',
      D => \AN[6]_i_1_n_0\,
      PRE => \AN[7]_i_2_n_0\,
      Q => AN(6)
    );
\AN_reg[7]\: unisim.vcomponents.FDPE
     port map (
      C => clk,
      CE => '1',
      D => \AN[7]_i_1_n_0\,
      PRE => \AN[7]_i_2_n_0\,
      Q => AN(7)
    );
\cnt[0]_i_1\: unisim.vcomponents.LUT2
    generic map(
      INIT => X"9"
    )
        port map (
      I0 => \cnt_ms[15]_i_2_n_0\,
      I1 => cnt(0),
      O => \cnt[0]_i_1_n_0\
    );
\cnt[1]_i_1\: unisim.vcomponents.LUT3
    generic map(
      INIT => X"D2"
    )
        port map (
      I0 => cnt(0),
      I1 => \cnt_ms[15]_i_2_n_0\,
      I2 => cnt(1),
      O => \cnt[1]_i_1_n_0\
    );
\cnt[2]_i_1\: unisim.vcomponents.LUT4
    generic map(
      INIT => X"F708"
    )
        port map (
      I0 => cnt(1),
      I1 => cnt(0),
      I2 => \cnt_ms[15]_i_2_n_0\,
      I3 => cnt(2),
      O => \cnt[2]_i_1_n_0\
    );
cnt_ms0_carry: unisim.vcomponents.CARRY4
     port map (
      CI => '0',
      CO(3) => cnt_ms0_carry_n_0,
      CO(2) => cnt_ms0_carry_n_1,
      CO(1) => cnt_ms0_carry_n_2,
      CO(0) => cnt_ms0_carry_n_3,
      CYINIT => cnt_ms(0),
      DI(3 downto 0) => B"0000",
      O(3 downto 0) => data0(4 downto 1),
      S(3 downto 0) => cnt_ms(4 downto 1)
    );
\cnt_ms0_carry__0\: unisim.vcomponents.CARRY4
     port map (
      CI => cnt_ms0_carry_n_0,
      CO(3) => \cnt_ms0_carry__0_n_0\,
      CO(2) => \cnt_ms0_carry__0_n_1\,
      CO(1) => \cnt_ms0_carry__0_n_2\,
      CO(0) => \cnt_ms0_carry__0_n_3\,
      CYINIT => '0',
      DI(3 downto 0) => B"0000",
      O(3 downto 0) => data0(8 downto 5),
      S(3 downto 0) => cnt_ms(8 downto 5)
    );
\cnt_ms0_carry__1\: unisim.vcomponents.CARRY4
     port map (
      CI => \cnt_ms0_carry__0_n_0\,
      CO(3) => \cnt_ms0_carry__1_n_0\,
      CO(2) => \cnt_ms0_carry__1_n_1\,
      CO(1) => \cnt_ms0_carry__1_n_2\,
      CO(0) => \cnt_ms0_carry__1_n_3\,
      CYINIT => '0',
      DI(3 downto 0) => B"0000",
      O(3 downto 0) => data0(12 downto 9),
      S(3 downto 0) => cnt_ms(12 downto 9)
    );
\cnt_ms0_carry__2\: unisim.vcomponents.CARRY4
     port map (
      CI => \cnt_ms0_carry__1_n_0\,
      CO(3 downto 2) => \NLW_cnt_ms0_carry__2_CO_UNCONNECTED\(3 downto 2),
      CO(1) => \cnt_ms0_carry__2_n_2\,
      CO(0) => \cnt_ms0_carry__2_n_3\,
      CYINIT => '0',
      DI(3 downto 0) => B"0000",
      O(3) => \NLW_cnt_ms0_carry__2_O_UNCONNECTED\(3),
      O(2 downto 0) => data0(15 downto 13),
      S(3) => '0',
      S(2 downto 0) => cnt_ms(15 downto 13)
    );
\cnt_ms[0]_i_1\: unisim.vcomponents.LUT1
    generic map(
      INIT => X"1"
    )
        port map (
      I0 => cnt_ms(0),
      O => cnt_ms_1(0)
    );
\cnt_ms[10]_i_1\: unisim.vcomponents.LUT2
    generic map(
      INIT => X"8"
    )
        port map (
      I0 => \cnt_ms[15]_i_2_n_0\,
      I1 => data0(10),
      O => cnt_ms_1(10)
    );
\cnt_ms[11]_i_1\: unisim.vcomponents.LUT2
    generic map(
      INIT => X"8"
    )
        port map (
      I0 => \cnt_ms[15]_i_2_n_0\,
      I1 => data0(11),
      O => cnt_ms_1(11)
    );
\cnt_ms[12]_i_1\: unisim.vcomponents.LUT2
    generic map(
      INIT => X"8"
    )
        port map (
      I0 => \cnt_ms[15]_i_2_n_0\,
      I1 => data0(12),
      O => cnt_ms_1(12)
    );
\cnt_ms[13]_i_1\: unisim.vcomponents.LUT2
    generic map(
      INIT => X"8"
    )
        port map (
      I0 => \cnt_ms[15]_i_2_n_0\,
      I1 => data0(13),
      O => cnt_ms_1(13)
    );
\cnt_ms[14]_i_1\: unisim.vcomponents.LUT2
    generic map(
      INIT => X"8"
    )
        port map (
      I0 => \cnt_ms[15]_i_2_n_0\,
      I1 => data0(14),
      O => cnt_ms_1(14)
    );
\cnt_ms[15]_i_1\: unisim.vcomponents.LUT2
    generic map(
      INIT => X"8"
    )
        port map (
      I0 => \cnt_ms[15]_i_2_n_0\,
      I1 => data0(15),
      O => cnt_ms_1(15)
    );
\cnt_ms[15]_i_2\: unisim.vcomponents.LUT4
    generic map(
      INIT => X"FFFE"
    )
        port map (
      I0 => \cnt_ms[15]_i_3_n_0\,
      I1 => \cnt_ms[15]_i_4_n_0\,
      I2 => \cnt_ms[15]_i_5_n_0\,
      I3 => \cnt_ms[15]_i_6_n_0\,
      O => \cnt_ms[15]_i_2_n_0\
    );
\cnt_ms[15]_i_3\: unisim.vcomponents.LUT4
    generic map(
      INIT => X"FFEF"
    )
        port map (
      I0 => cnt_ms(5),
      I1 => cnt_ms(4),
      I2 => cnt_ms(6),
      I3 => cnt_ms(7),
      O => \cnt_ms[15]_i_3_n_0\
    );
\cnt_ms[15]_i_4\: unisim.vcomponents.LUT4
    generic map(
      INIT => X"7FFF"
    )
        port map (
      I0 => cnt_ms(1),
      I1 => cnt_ms(0),
      I2 => cnt_ms(3),
      I3 => cnt_ms(2),
      O => \cnt_ms[15]_i_4_n_0\
    );
\cnt_ms[15]_i_5\: unisim.vcomponents.LUT4
    generic map(
      INIT => X"EFFF"
    )
        port map (
      I0 => cnt_ms(13),
      I1 => cnt_ms(12),
      I2 => cnt_ms(15),
      I3 => cnt_ms(14),
      O => \cnt_ms[15]_i_5_n_0\
    );
\cnt_ms[15]_i_6\: unisim.vcomponents.LUT4
    generic map(
      INIT => X"FFF7"
    )
        port map (
      I0 => cnt_ms(9),
      I1 => cnt_ms(8),
      I2 => cnt_ms(11),
      I3 => cnt_ms(10),
      O => \cnt_ms[15]_i_6_n_0\
    );
\cnt_ms[1]_i_1\: unisim.vcomponents.LUT2
    generic map(
      INIT => X"8"
    )
        port map (
      I0 => \cnt_ms[15]_i_2_n_0\,
      I1 => data0(1),
      O => cnt_ms_1(1)
    );
\cnt_ms[2]_i_1\: unisim.vcomponents.LUT2
    generic map(
      INIT => X"8"
    )
        port map (
      I0 => \cnt_ms[15]_i_2_n_0\,
      I1 => data0(2),
      O => cnt_ms_1(2)
    );
\cnt_ms[3]_i_1\: unisim.vcomponents.LUT2
    generic map(
      INIT => X"8"
    )
        port map (
      I0 => \cnt_ms[15]_i_2_n_0\,
      I1 => data0(3),
      O => cnt_ms_1(3)
    );
\cnt_ms[4]_i_1\: unisim.vcomponents.LUT2
    generic map(
      INIT => X"8"
    )
        port map (
      I0 => \cnt_ms[15]_i_2_n_0\,
      I1 => data0(4),
      O => cnt_ms_1(4)
    );
\cnt_ms[5]_i_1\: unisim.vcomponents.LUT2
    generic map(
      INIT => X"8"
    )
        port map (
      I0 => \cnt_ms[15]_i_2_n_0\,
      I1 => data0(5),
      O => cnt_ms_1(5)
    );
\cnt_ms[6]_i_1\: unisim.vcomponents.LUT2
    generic map(
      INIT => X"8"
    )
        port map (
      I0 => \cnt_ms[15]_i_2_n_0\,
      I1 => data0(6),
      O => cnt_ms_1(6)
    );
\cnt_ms[7]_i_1\: unisim.vcomponents.LUT2
    generic map(
      INIT => X"8"
    )
        port map (
      I0 => \cnt_ms[15]_i_2_n_0\,
      I1 => data0(7),
      O => cnt_ms_1(7)
    );
\cnt_ms[8]_i_1\: unisim.vcomponents.LUT2
    generic map(
      INIT => X"8"
    )
        port map (
      I0 => \cnt_ms[15]_i_2_n_0\,
      I1 => data0(8),
      O => cnt_ms_1(8)
    );
\cnt_ms[9]_i_1\: unisim.vcomponents.LUT2
    generic map(
      INIT => X"8"
    )
        port map (
      I0 => \cnt_ms[15]_i_2_n_0\,
      I1 => data0(9),
      O => cnt_ms_1(9)
    );
\cnt_ms_reg[0]\: unisim.vcomponents.FDCE
     port map (
      C => clk,
      CE => '1',
      CLR => \AN[7]_i_2_n_0\,
      D => cnt_ms_1(0),
      Q => cnt_ms(0)
    );
\cnt_ms_reg[10]\: unisim.vcomponents.FDCE
     port map (
      C => clk,
      CE => '1',
      CLR => \AN[7]_i_2_n_0\,
      D => cnt_ms_1(10),
      Q => cnt_ms(10)
    );
\cnt_ms_reg[11]\: unisim.vcomponents.FDCE
     port map (
      C => clk,
      CE => '1',
      CLR => \AN[7]_i_2_n_0\,
      D => cnt_ms_1(11),
      Q => cnt_ms(11)
    );
\cnt_ms_reg[12]\: unisim.vcomponents.FDCE
     port map (
      C => clk,
      CE => '1',
      CLR => \AN[7]_i_2_n_0\,
      D => cnt_ms_1(12),
      Q => cnt_ms(12)
    );
\cnt_ms_reg[13]\: unisim.vcomponents.FDCE
     port map (
      C => clk,
      CE => '1',
      CLR => \AN[7]_i_2_n_0\,
      D => cnt_ms_1(13),
      Q => cnt_ms(13)
    );
\cnt_ms_reg[14]\: unisim.vcomponents.FDCE
     port map (
      C => clk,
      CE => '1',
      CLR => \AN[7]_i_2_n_0\,
      D => cnt_ms_1(14),
      Q => cnt_ms(14)
    );
\cnt_ms_reg[15]\: unisim.vcomponents.FDCE
     port map (
      C => clk,
      CE => '1',
      CLR => \AN[7]_i_2_n_0\,
      D => cnt_ms_1(15),
      Q => cnt_ms(15)
    );
\cnt_ms_reg[1]\: unisim.vcomponents.FDCE
     port map (
      C => clk,
      CE => '1',
      CLR => \AN[7]_i_2_n_0\,
      D => cnt_ms_1(1),
      Q => cnt_ms(1)
    );
\cnt_ms_reg[2]\: unisim.vcomponents.FDCE
     port map (
      C => clk,
      CE => '1',
      CLR => \AN[7]_i_2_n_0\,
      D => cnt_ms_1(2),
      Q => cnt_ms(2)
    );
\cnt_ms_reg[3]\: unisim.vcomponents.FDCE
     port map (
      C => clk,
      CE => '1',
      CLR => \AN[7]_i_2_n_0\,
      D => cnt_ms_1(3),
      Q => cnt_ms(3)
    );
\cnt_ms_reg[4]\: unisim.vcomponents.FDCE
     port map (
      C => clk,
      CE => '1',
      CLR => \AN[7]_i_2_n_0\,
      D => cnt_ms_1(4),
      Q => cnt_ms(4)
    );
\cnt_ms_reg[5]\: unisim.vcomponents.FDCE
     port map (
      C => clk,
      CE => '1',
      CLR => \AN[7]_i_2_n_0\,
      D => cnt_ms_1(5),
      Q => cnt_ms(5)
    );
\cnt_ms_reg[6]\: unisim.vcomponents.FDCE
     port map (
      C => clk,
      CE => '1',
      CLR => \AN[7]_i_2_n_0\,
      D => cnt_ms_1(6),
      Q => cnt_ms(6)
    );
\cnt_ms_reg[7]\: unisim.vcomponents.FDCE
     port map (
      C => clk,
      CE => '1',
      CLR => \AN[7]_i_2_n_0\,
      D => cnt_ms_1(7),
      Q => cnt_ms(7)
    );
\cnt_ms_reg[8]\: unisim.vcomponents.FDCE
     port map (
      C => clk,
      CE => '1',
      CLR => \AN[7]_i_2_n_0\,
      D => cnt_ms_1(8),
      Q => cnt_ms(8)
    );
\cnt_ms_reg[9]\: unisim.vcomponents.FDCE
     port map (
      C => clk,
      CE => '1',
      CLR => \AN[7]_i_2_n_0\,
      D => cnt_ms_1(9),
      Q => cnt_ms(9)
    );
\cnt_reg[0]\: unisim.vcomponents.FDCE
     port map (
      C => clk,
      CE => '1',
      CLR => \AN[7]_i_2_n_0\,
      D => \cnt[0]_i_1_n_0\,
      Q => cnt(0)
    );
\cnt_reg[1]\: unisim.vcomponents.FDCE
     port map (
      C => clk,
      CE => '1',
      CLR => \AN[7]_i_2_n_0\,
      D => \cnt[1]_i_1_n_0\,
      Q => cnt(1)
    );
\cnt_reg[2]\: unisim.vcomponents.FDCE
     port map (
      C => clk,
      CE => '1',
      CLR => \AN[7]_i_2_n_0\,
      D => \cnt[2]_i_1_n_0\,
      Q => cnt(2)
    );
\seg_data_o[0]_i_1\: unisim.vcomponents.LUT4
    generic map(
      INIT => X"2094"
    )
        port map (
      I0 => seg_num(3),
      I1 => seg_num(2),
      I2 => seg_num(0),
      I3 => seg_num(1),
      O => seg_data_o_0(0)
    );
\seg_data_o[1]_i_1\: unisim.vcomponents.LUT4
    generic map(
      INIT => X"A4C8"
    )
        port map (
      I0 => seg_num(3),
      I1 => seg_num(2),
      I2 => seg_num(1),
      I3 => seg_num(0),
      O => seg_data_o_0(1)
    );
\seg_data_o[2]_i_1\: unisim.vcomponents.LUT4
    generic map(
      INIT => X"A210"
    )
        port map (
      I0 => seg_num(3),
      I1 => seg_num(0),
      I2 => seg_num(1),
      I3 => seg_num(2),
      O => seg_data_o_0(2)
    );
\seg_data_o[3]_i_1\: unisim.vcomponents.LUT4
    generic map(
      INIT => X"C214"
    )
        port map (
      I0 => seg_num(3),
      I1 => seg_num(2),
      I2 => seg_num(0),
      I3 => seg_num(1),
      O => seg_data_o_0(3)
    );
\seg_data_o[4]_i_1\: unisim.vcomponents.LUT4
    generic map(
      INIT => X"5710"
    )
        port map (
      I0 => seg_num(3),
      I1 => seg_num(1),
      I2 => seg_num(2),
      I3 => seg_num(0),
      O => seg_data_o_0(4)
    );
\seg_data_o[5]_i_1\: unisim.vcomponents.LUT4
    generic map(
      INIT => X"5190"
    )
        port map (
      I0 => seg_num(3),
      I1 => seg_num(2),
      I2 => seg_num(0),
      I3 => seg_num(1),
      O => seg_data_o_0(5)
    );
\seg_data_o[6]_i_1\: unisim.vcomponents.LUT4
    generic map(
      INIT => X"4025"
    )
        port map (
      I0 => seg_num(3),
      I1 => seg_num(0),
      I2 => seg_num(2),
      I3 => seg_num(1),
      O => seg_data_o_0(6)
    );
\seg_data_o_reg[0]\: unisim.vcomponents.FDPE
     port map (
      C => clk,
      CE => '1',
      D => seg_data_o_0(0),
      PRE => \AN[7]_i_2_n_0\,
      Q => seg_data_o(0)
    );
\seg_data_o_reg[1]\: unisim.vcomponents.FDPE
     port map (
      C => clk,
      CE => '1',
      D => seg_data_o_0(1),
      PRE => \AN[7]_i_2_n_0\,
      Q => seg_data_o(1)
    );
\seg_data_o_reg[2]\: unisim.vcomponents.FDPE
     port map (
      C => clk,
      CE => '1',
      D => seg_data_o_0(2),
      PRE => \AN[7]_i_2_n_0\,
      Q => seg_data_o(2)
    );
\seg_data_o_reg[3]\: unisim.vcomponents.FDPE
     port map (
      C => clk,
      CE => '1',
      D => seg_data_o_0(3),
      PRE => \AN[7]_i_2_n_0\,
      Q => seg_data_o(3)
    );
\seg_data_o_reg[4]\: unisim.vcomponents.FDPE
     port map (
      C => clk,
      CE => '1',
      D => seg_data_o_0(4),
      PRE => \AN[7]_i_2_n_0\,
      Q => seg_data_o(4)
    );
\seg_data_o_reg[5]\: unisim.vcomponents.FDPE
     port map (
      C => clk,
      CE => '1',
      D => seg_data_o_0(5),
      PRE => \AN[7]_i_2_n_0\,
      Q => seg_data_o(5)
    );
\seg_data_o_reg[6]\: unisim.vcomponents.FDPE
     port map (
      C => clk,
      CE => '1',
      D => seg_data_o_0(6),
      PRE => \AN[7]_i_2_n_0\,
      Q => seg_data_o(6)
    );
\seg_num[0]_i_2\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"AFA0CFCFAFA0C0C0"
    )
        port map (
      I0 => Data_i(12),
      I1 => Data_i(8),
      I2 => cnt(1),
      I3 => Data_i(4),
      I4 => cnt(0),
      I5 => Data_i(0),
      O => \seg_num[0]_i_2_n_0\
    );
\seg_num[0]_i_3\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"AFA0CFCFAFA0C0C0"
    )
        port map (
      I0 => Data_i(28),
      I1 => Data_i(24),
      I2 => cnt(1),
      I3 => Data_i(20),
      I4 => cnt(0),
      I5 => Data_i(16),
      O => \seg_num[0]_i_3_n_0\
    );
\seg_num[1]_i_2\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"AFA0CFCFAFA0C0C0"
    )
        port map (
      I0 => Data_i(13),
      I1 => Data_i(9),
      I2 => cnt(1),
      I3 => Data_i(5),
      I4 => cnt(0),
      I5 => Data_i(1),
      O => \seg_num[1]_i_2_n_0\
    );
\seg_num[1]_i_3\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"AFA0CFCFAFA0C0C0"
    )
        port map (
      I0 => Data_i(29),
      I1 => Data_i(25),
      I2 => cnt(1),
      I3 => Data_i(21),
      I4 => cnt(0),
      I5 => Data_i(17),
      O => \seg_num[1]_i_3_n_0\
    );
\seg_num[2]_i_2\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"AFA0CFCFAFA0C0C0"
    )
        port map (
      I0 => Data_i(14),
      I1 => Data_i(10),
      I2 => cnt(1),
      I3 => Data_i(6),
      I4 => cnt(0),
      I5 => Data_i(2),
      O => \seg_num[2]_i_2_n_0\
    );
\seg_num[2]_i_3\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"AFA0CFCFAFA0C0C0"
    )
        port map (
      I0 => Data_i(30),
      I1 => Data_i(26),
      I2 => cnt(1),
      I3 => Data_i(22),
      I4 => cnt(0),
      I5 => Data_i(18),
      O => \seg_num[2]_i_3_n_0\
    );
\seg_num[3]_i_2\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"AFA0CFCFAFA0C0C0"
    )
        port map (
      I0 => Data_i(15),
      I1 => Data_i(11),
      I2 => cnt(1),
      I3 => Data_i(7),
      I4 => cnt(0),
      I5 => Data_i(3),
      O => \seg_num[3]_i_2_n_0\
    );
\seg_num[3]_i_3\: unisim.vcomponents.LUT6
    generic map(
      INIT => X"AFA0CFCFAFA0C0C0"
    )
        port map (
      I0 => Data_i(31),
      I1 => Data_i(27),
      I2 => cnt(1),
      I3 => Data_i(23),
      I4 => cnt(0),
      I5 => Data_i(19),
      O => \seg_num[3]_i_3_n_0\
    );
\seg_num_reg[0]\: unisim.vcomponents.FDCE
     port map (
      C => clk,
      CE => '1',
      CLR => \AN[7]_i_2_n_0\,
      D => seg_num_2(0),
      Q => seg_num(0)
    );
\seg_num_reg[0]_i_1\: unisim.vcomponents.MUXF7
     port map (
      I0 => \seg_num[0]_i_2_n_0\,
      I1 => \seg_num[0]_i_3_n_0\,
      O => seg_num_2(0),
      S => cnt(2)
    );
\seg_num_reg[1]\: unisim.vcomponents.FDCE
     port map (
      C => clk,
      CE => '1',
      CLR => \AN[7]_i_2_n_0\,
      D => seg_num_2(1),
      Q => seg_num(1)
    );
\seg_num_reg[1]_i_1\: unisim.vcomponents.MUXF7
     port map (
      I0 => \seg_num[1]_i_2_n_0\,
      I1 => \seg_num[1]_i_3_n_0\,
      O => seg_num_2(1),
      S => cnt(2)
    );
\seg_num_reg[2]\: unisim.vcomponents.FDCE
     port map (
      C => clk,
      CE => '1',
      CLR => \AN[7]_i_2_n_0\,
      D => seg_num_2(2),
      Q => seg_num(2)
    );
\seg_num_reg[2]_i_1\: unisim.vcomponents.MUXF7
     port map (
      I0 => \seg_num[2]_i_2_n_0\,
      I1 => \seg_num[2]_i_3_n_0\,
      O => seg_num_2(2),
      S => cnt(2)
    );
\seg_num_reg[3]\: unisim.vcomponents.FDCE
     port map (
      C => clk,
      CE => '1',
      CLR => \AN[7]_i_2_n_0\,
      D => seg_num_2(3),
      Q => seg_num(3)
    );
\seg_num_reg[3]_i_1\: unisim.vcomponents.MUXF7
     port map (
      I0 => \seg_num[3]_i_2_n_0\,
      I1 => \seg_num[3]_i_3_n_0\,
      O => seg_num_2(3),
      S => cnt(2)
    );
end STRUCTURE;
library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
library UNISIM;
use UNISIM.VCOMPONENTS.ALL;
entity risc32_segment_0_0 is
  port (
    clk : in STD_LOGIC;
    rst_n : in STD_LOGIC;
    Data_i : in STD_LOGIC_VECTOR ( 31 downto 0 );
    AN : out STD_LOGIC_VECTOR ( 7 downto 0 );
    seg_data_o : out STD_LOGIC_VECTOR ( 7 downto 0 )
  );
  attribute NotValidForBitStream : boolean;
  attribute NotValidForBitStream of risc32_segment_0_0 : entity is true;
  attribute CHECK_LICENSE_TYPE : string;
  attribute CHECK_LICENSE_TYPE of risc32_segment_0_0 : entity is "risc32_segment_0_0,segment,{}";
  attribute DowngradeIPIdentifiedWarnings : string;
  attribute DowngradeIPIdentifiedWarnings of risc32_segment_0_0 : entity is "yes";
  attribute IP_DEFINITION_SOURCE : string;
  attribute IP_DEFINITION_SOURCE of risc32_segment_0_0 : entity is "module_ref";
  attribute X_CORE_INFO : string;
  attribute X_CORE_INFO of risc32_segment_0_0 : entity is "segment,Vivado 2022.2";
end risc32_segment_0_0;

architecture STRUCTURE of risc32_segment_0_0 is
  signal \<const1>\ : STD_LOGIC;
  signal \^seg_data_o\ : STD_LOGIC_VECTOR ( 6 downto 0 );
  attribute X_INTERFACE_INFO : string;
  attribute X_INTERFACE_INFO of clk : signal is "xilinx.com:signal:clock:1.0 clk CLK";
  attribute X_INTERFACE_PARAMETER : string;
  attribute X_INTERFACE_PARAMETER of clk : signal is "XIL_INTERFACENAME clk, FREQ_HZ 50000000, FREQ_TOLERANCE_HZ 0, PHASE 0.0, CLK_DOMAIN /clk_wiz_0_clk_out1, INSERT_VIP 0";
  attribute X_INTERFACE_INFO of rst_n : signal is "xilinx.com:signal:reset:1.0 rst_n RST";
  attribute X_INTERFACE_PARAMETER of rst_n : signal is "XIL_INTERFACENAME rst_n, POLARITY ACTIVE_LOW, INSERT_VIP 0";
begin
  seg_data_o(7) <= \<const1>\;
  seg_data_o(6 downto 0) <= \^seg_data_o\(6 downto 0);
VCC: unisim.vcomponents.VCC
     port map (
      P => \<const1>\
    );
inst: entity work.risc32_segment_0_0_segment
     port map (
      AN(7 downto 0) => AN(7 downto 0),
      Data_i(31 downto 0) => Data_i(31 downto 0),
      clk => clk,
      rst_n => rst_n,
      seg_data_o(6 downto 0) => \^seg_data_o\(6 downto 0)
    );
end STRUCTURE;
