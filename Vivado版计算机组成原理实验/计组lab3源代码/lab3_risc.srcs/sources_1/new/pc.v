`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: lzut
// Engineer: Yao bin
// 
// Create Date: 2024/10/19 16:06:29
// Design Name: 
// Module Name: pc
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


module pc(  input               clk     ,
            input               pc_clr  ,
            input       [31:0]  data_i  ,
            output  reg [31:0]  q       );

always @(posedge clk or negedge pc_clr)
    begin
        if(!pc_clr)
            q   <=  32'd0;
        else
            q   <=  data_i;       
    end

endmodule
