`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: lzut
// Engineer: Yao bin
// 
// Create Date: 2024/09/22 09:57:00
// Design Name: 
// Module Name: segment
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


module segment
(
    input                       clk         ,           //50MHz
    input                       rst_n       ,
    input           [31:0]      Data_i      ,
    
    output  reg     [7:0]       AN          ,           //位选
    output  reg     [7:0]       seg_data_o              //段选
);

parameter           zero        =   8'b1100_0000,
                    one         =   8'b1111_1001,
                    two         =   8'b1010_0100,
                    three       =   8'b1011_0000,
                    four        =   8'b1001_1001,
                    five        =   8'b1001_0010,
                    six         =   8'b1000_0010,
                    seven       =   8'b1111_1000,
                    eight       =   8'b1000_0000,
                    nine        =   8'b1001_0000,
                    a_display   =   8'b1000_1000,
                    b_display   =   8'b1000_0011,
                    c_display   =   8'b1100_0110,
                    d_display   =   8'b1010_0001,
                    e_display   =   8'b1000_0110,
                    f_display   =   8'b1000_1110,
                    
                    WIDTH       =   16'd4_9999      ;

reg     [15:0]      cnt_ms  ;               //1ms计数
reg     [2:0]       cnt     ;               //数码管编号
reg     [3:0]       seg_num ;               //段选数字     
  
always @(posedge clk or negedge rst_n)
    begin
        if(rst_n == 1'b0)
            cnt_ms <= 15'd0;
        else if(cnt_ms == WIDTH)
            cnt_ms <= 15'b0;
        else
            cnt_ms <= cnt_ms + 1'b1;   
    end    

always @(posedge clk or negedge rst_n)
    begin
        if(rst_n == 1'b0)
            cnt <= 3'd0;
        else if(cnt_ms == WIDTH)
            cnt <= cnt + 1'b1;
        else
            cnt <= cnt;    
    end

//数码管位选、段选数值    
always @(posedge clk or negedge rst_n)
    begin
        if(rst_n == 1'b0)
            begin
                AN <= 8'b1111_1111;
                seg_num <= 4'b0;
            end
        else
            begin
                case (cnt)
                    3'd0 : begin
                                AN <= 8'b1111_1110;
                                seg_num <= Data_i[3:0];
                           end
                    3'd1 : begin
                                AN <= 8'b1111_1101;
                                seg_num <= Data_i[7:4]; 
                           end
                    3'd2 : begin
                                AN <= 8'b1111_1011;
                                seg_num <= Data_i[11:8];                                
                           end
                    3'd3 : begin
                                AN <= 8'b1111_0111;
                                seg_num <= Data_i[15:12];
                           end
                    3'd4 : begin
                                AN <= 8'b1110_1111;
                                seg_num <= Data_i[19:16];
                           end
                    3'd5 : begin
                                AN <= 8'b1101_1111;
                                seg_num <= Data_i[23:20];
                           end
                    3'd6 : begin
                                AN <= 8'b1011_1111;
                                seg_num <= Data_i[27:24];
                           end
                    3'd7 : begin
                                AN <= 8'b0111_1111;
                                seg_num <= Data_i[31:28];
                           end
                           
                    default : begin
                                AN <= 8'b1111_1111;
                                seg_num <= 4'b0000;
                              end
                endcase            
            end    
    end
    
//数码管段选值
always @(posedge clk or negedge rst_n)
    begin
        if(rst_n == 1'b0)
            seg_data_o <= 8'b1111_1111;
        else begin
            case (seg_num)
                4'd0 : seg_data_o   <=  zero       ;
                4'd1 : seg_data_o   <=  one        ;
                4'd2 : seg_data_o   <=  two        ;
                4'd3 : seg_data_o   <=  three      ;
                4'd4 : seg_data_o   <=  four       ;
                4'd5 : seg_data_o   <=  five       ;
                4'd6 : seg_data_o   <=  six        ;
                4'd7 : seg_data_o   <=  seven      ;
                4'd8 : seg_data_o   <=  eight      ;
                4'd9 : seg_data_o   <=  nine       ;
                4'ha : seg_data_o   <=  a_display  ;
                4'hb : seg_data_o   <=  b_display  ;
                4'hc : seg_data_o   <=  c_display  ;
                4'hd : seg_data_o   <=  d_display  ;
                4'he : seg_data_o   <=  e_display  ;
                4'hf : seg_data_o   <=  f_display  ;                
            endcase
        end 
    end

endmodule

