"""
Runs every section and summarises what passes and what does not.

Run with: pdm run python -m project_name.sailing.testing.run_all
"""

import sys

from project_name.sailing.testing import (angle_testing,
                                          env_testing,
                                          foil_testing,
                                          manoeuvre_testing,
                                          polar_testing,
                                          rigid_body_testing,
                                          stability_testing,
                                          tack_testing,
                                          )


if __name__ == "__main__":
    sections = [module.main() for module in (angle_testing,
                                             foil_testing,
                                             rigid_body_testing,
                                             stability_testing,
                                             polar_testing,
                                             manoeuvre_testing,
                                             tack_testing,
                                             env_testing)]

    print("\n=== Summary ===")
    for section in sections:
        print(f"  {len(section.results) - len(section.failed):2d}/{len(section.results):2d}  {section.section}")
        for description in section.failed:
            print(f"         FAIL: {description}")
    sys.exit(1 if any(section.failed for section in sections) else 0)
