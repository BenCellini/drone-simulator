# drone-simulator
Simulation &amp; control of a dynamical model of a quadcopter drone. Also evaluate the empirical observability of each drone state

## Example
Start with a simple noebook example: [drone_example.ipynb](drone_example.ipynb)


## Variable names
| Variable     | meaning                                                               |
|--------------|-----------------------------------------------------------------------|
| $x$          | x-position                                                            |
| $y$          | y-position                                                            |
| $z$          | z-position                                                            |
| $\phi$       | roll angle [rad]                                                      |
| $\theta$     | pitch angle [rad]                                                     |
| $\psi$       | yaw angle (heading) [rad]                                             |
| $\omega_x$   | angular velocity about x-axis (roll) [rad/s]                          |
| $\omega_y$   | angular velocity about y-axis (roll) [rad/s]                          |
| $\omega_z$   | angular velocity about z-axis (yaw) [rad/s]                           |
| $v_x$        | velocity parallel to heading in body-level frame                      |
| $v_y$        | velocity perpendicular to heading in body-level frame                 |
| $v_z$        | vertical velocity in body-level frame                                 |
| $\dot{v}_x$  | acceleration parallel to heading in body-level frame                  |
| $\dot{v}_y$  | acceleration perpendicular to heading in body-level frame             |
| $\dot{v}_z$  | vertical acceleration in body-level frame                             |
| $q$          | acceleration magnitude in horizontal body-level frame                 |
| $\alpha$     | acceleration angle in horizontal body-level frame [rad]               |
| $u_{thrust}$ | vertical thrust in body frame                                         |
| $u_{\phi}$   | roll torque                                                           |
| $u_{\theta}$ | pitch torque                                                          |
| $u_{\psi}$   | yaw torque                                                            |
| $u_{x}$      | projected thrust parallel to heading in body-level frame              |
| $u_{y}$      | projected thrust perpendicular to heading in body-level frame         |
| $w$          | ambient wind speed                                                    |
| $\zeta$      | ambient wind direction [rad]                                          |
| $r_x$        | optic flow parallel to heading in body-level frame                    |
| $r_y$        | optic flow perpendicular to heading in body-level frame               |
| $r$          | optic flow magnitude in horizontal body-level frame                   |
| $g$          | ground speed magnitude in horizontal body-level frame                 |
| $\beta$      | ground speed (&optic flow) angle in horizontal body-level frame [rad] |
| $a_x$        | apparent airflow parallel to heading in body-level frame              |
| $a_y$        | apparent airflow perpendicular to heading in body-level frame         |
| $a$          | apparent airflow magnitude in horizontal body-level frame             |
| $\gamma$     | apparent airflow angle in horizontal body-level frame [rad]           |
| $m$          | mass                                                                  |
| $I_x$        | mass-moment of inertia about x-axis                                   |
| $I_y$        | mass-moment of inertia about y-axis                                   |
| $I_z$        | mass-moment of inertia about z-axis                                   |
| $C$          | translation damping coefficient                                       |


