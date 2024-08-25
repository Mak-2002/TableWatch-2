from monitoring_system import RestaurantMonitoringSystem

if __name__ == "__main__":
    restaurant_monitor = RestaurantMonitoringSystem(playback_speed=1.0, display_detection_logs=False)
    restaurant_monitor.run()
