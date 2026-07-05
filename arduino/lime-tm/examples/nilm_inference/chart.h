#ifndef __LV_CHART_H__
#define __LV_CHART_H__

#include <lvgl.h>
#include <TFT_eSPI.h>

#include <examples/lv_examples.h>
#include <demos/lv_demos.h>

#define LV_CHART_POINT 30

/*Set to your screen resolution and rotation*/
#define TFT_HOR_RES 240
#define TFT_VER_RES 320
#define TFT_ROTATION LV_DISPLAY_ROTATION_90

/*LVGL draw into this buffer, 1/10 screen size usually works well. The size is in bytes*/
#define DRAW_BUF_SIZE (TFT_HOR_RES * TFT_VER_RES / 10 * (LV_COLOR_DEPTH / 8))

void lv_chart_ui();
void lv_ui_init();
void lv_update_chart(float value);

#endif __LV_CHART_H__
