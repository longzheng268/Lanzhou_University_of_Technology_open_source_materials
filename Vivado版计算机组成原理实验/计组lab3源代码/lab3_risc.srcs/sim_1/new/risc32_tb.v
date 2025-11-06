`timescale 1ns / 1ns
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2024/10/28 16:10:28
// Design Name: 
// Module Name: risc32_tb
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////


module risc32_tb();
reg             pc_clr;
reg             clk;
wire    [31:0]  alu_out;

initial
    begin
        clk     =   1'b0;
        pc_clr  <=  1'b0;        
        #20
        pc_clr  <=  1'b1;
    end
always  #10 clk =   ~clk;   

risc32_wrapper  inst1
(
    .alu_out_0  (alu_out)   ,
    .clk        (clk)       ,
    .pc_clr     (pc_clr)    
 );

endmodule
