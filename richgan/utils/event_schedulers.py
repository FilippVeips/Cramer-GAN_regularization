class PeriodicScheduler:
    def __init__(self):
        self.passthrough = False

    def set_passthrough(self):
        self.passthrough = True

    def unset_passthrough(self):
        self.passthrough = False

    def check_step(self, step, period):
        if self.passthrough:
            return True
        return (step + 1) % period == 0


global_periodic_scheduler = PeriodicScheduler()
global_check_step = global_periodic_scheduler.check_step
