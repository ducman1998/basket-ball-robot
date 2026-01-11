/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.h
  * @brief          : Header for main.c file.
  *                   This file contains the common defines of the application.
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2025 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */
/* USER CODE END Header */

/* Define to prevent recursive inclusion -------------------------------------*/
#ifndef __MAIN_H
#define __MAIN_H

#ifdef __cplusplus
extern "C" {
#endif

/* Includes ------------------------------------------------------------------*/
#include "stm32g4xx_hal.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */

/* USER CODE END Includes */

/* Exported types ------------------------------------------------------------*/
/* USER CODE BEGIN ET */

/* USER CODE END ET */

/* Exported constants --------------------------------------------------------*/
/* USER CODE BEGIN EC */

/* USER CODE END EC */

/* Exported macro ------------------------------------------------------------*/
/* USER CODE BEGIN EM */

/* USER CODE END EM */

void HAL_TIM_MspPostInit(TIM_HandleTypeDef *htim);

/* Exported functions prototypes ---------------------------------------------*/
void Error_Handler(void);

/* USER CODE BEGIN EFP */

/* USER CODE END EFP */

/* Private defines -----------------------------------------------------------*/
#define IR_RX_Pin GPIO_PIN_1
#define IR_RX_GPIO_Port GPIOF
#define E_M1_A_Pin GPIO_PIN_0
#define E_M1_A_GPIO_Port GPIOA
#define E_M1_B_Pin GPIO_PIN_1
#define E_M1_B_GPIO_Port GPIOA
#define M1_DIR_Pin GPIO_PIN_2
#define M1_DIR_GPIO_Port GPIOA
#define M2_DIR_Pin GPIO_PIN_3
#define M2_DIR_GPIO_Port GPIOA
#define E_M2_B_Pin GPIO_PIN_4
#define E_M2_B_GPIO_Port GPIOA
#define M3_DIR_Pin GPIO_PIN_5
#define M3_DIR_GPIO_Port GPIOA
#define E_M2_A_Pin GPIO_PIN_6
#define E_M2_A_GPIO_Port GPIOA
#define NSLEEP_Pin GPIO_PIN_7
#define NSLEEP_GPIO_Port GPIOA
#define PWM_M1_Pin GPIO_PIN_8
#define PWM_M1_GPIO_Port GPIOA
#define PWM_M2_Pin GPIO_PIN_9
#define PWM_M2_GPIO_Port GPIOA
#define PWM_M3_Pin GPIO_PIN_10
#define PWM_M3_GPIO_Port GPIOA
#define PWM_TH_Pin GPIO_PIN_15
#define PWM_TH_GPIO_Port GPIOA
#define MCU_LED_Pin GPIO_PIN_3
#define MCU_LED_GPIO_Port GPIOB
#define PWM_GRB_Pin GPIO_PIN_4
#define PWM_GRB_GPIO_Port GPIOB
#define PWM_ANG_Pin GPIO_PIN_5
#define PWM_ANG_GPIO_Port GPIOB
#define E_M3_A_Pin GPIO_PIN_6
#define E_M3_A_GPIO_Port GPIOB
#define E_M3_B_Pin GPIO_PIN_7
#define E_M3_B_GPIO_Port GPIOB

/* USER CODE BEGIN Private defines */

/* USER CODE END Private defines */

#ifdef __cplusplus
}
#endif

#endif /* __MAIN_H */
