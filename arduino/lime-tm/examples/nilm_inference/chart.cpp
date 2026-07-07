#include "chart.h"

lv_obj_t* value_label = nullptr;

lv_obj_t* power_chart = nullptr;
lv_chart_series_t* power_series = nullptr;

float chart_min = 0.0f;
float chart_max = 1000.0f;

uint8_t* draw_buf;      //draw_buf is allocated on heap otherwise the static area is too big on ESP32 at compile

void lv_ui_init() {
  lv_init();
  draw_buf = new uint8_t[DRAW_BUF_SIZE];
  lv_display_t* disp;
  disp = lv_tft_espi_create(TFT_HOR_RES, TFT_VER_RES, draw_buf, DRAW_BUF_SIZE);
  lv_display_set_rotation(disp, TFT_ROTATION);
}

#if LV_USE_LOG != 0
void my_print(lv_log_level_t level, const char* buf) {
  LV_UNUSED(level);
  Serial.println(buf);
  Serial.flush();
}
#endif

/* LVGL calls it when a rendered image needs to copied to the display*/
void my_disp_flush(lv_display_t* disp, const lv_area_t* area, uint8_t* px_map) {
  /*Call it to tell LVGL you are ready*/
  lv_disp_flush_ready(disp);
}

void lv_chart_ui() {
  lv_obj_t* scr = lv_scr_act();

  value_label = lv_label_create(scr);
  lv_label_set_text(value_label, "Value: -- | R: 0 F: 0 M: 0");
  lv_obj_align(value_label, LV_ALIGN_TOP_MID, 0, 8);

  power_chart = lv_chart_create(scr);
  lv_obj_set_size(power_chart, TFT_VER_RES, TFT_HOR_RES - 30);
  lv_obj_align(power_chart, LV_ALIGN_BOTTOM_MID, 0, 0);
  lv_chart_set_type(power_chart, LV_CHART_TYPE_LINE);
  lv_chart_set_point_count(power_chart, LV_CHART_POINT);
  lv_chart_set_range(power_chart, LV_CHART_AXIS_PRIMARY_Y, (int32_t)chart_min, (int32_t)chart_max);

  // lv_chart_set_div_line_count(power_chart, 5, 6);

  power_series = lv_chart_add_series(power_chart, lv_palette_main(LV_PALETTE_GREEN), LV_CHART_AXIS_PRIMARY_Y);
}

void lv_update_chart(float value, size_t rising_count, size_t falling_count, size_t matched_count) {
  if (value < chart_min) {
    chart_min = value;
    lv_chart_set_range(power_chart, LV_CHART_AXIS_PRIMARY_Y, (int32_t)chart_min, (int32_t)chart_max);
  }
  if (value > chart_max) {
    chart_max = value;
    lv_chart_set_range(power_chart, LV_CHART_AXIS_PRIMARY_Y, (int32_t)chart_min, (int32_t)chart_max);
  }

  lv_chart_set_next_value(power_chart, power_series, (lv_coord_t)value);

  char label_text[64];
  snprintf(
      label_text,
      sizeof(label_text),
      "Value: %.2f | R: %u F: %u M: %u",
      value,
      (unsigned int)rising_count,
      (unsigned int)falling_count,
      (unsigned int)matched_count);
  lv_label_set_text(value_label, label_text);
}
