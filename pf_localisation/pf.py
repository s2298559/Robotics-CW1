from geometry_msgs.msg import Pose, PoseArray, Quaternion, Point
from . pf_base import PFLocaliserBase

from . util import rotateQuaternion, getHeading


class PFLocaliser(PFLocaliserBase):
       
    def __init__(self):
        # ----- Call the superclass constructor
        super(PFLocaliser, self).__init__()
        
        # ----- Set motion model parameters
        self.ODOM_ROTATION_NOISE = 0.05
        self.ODOM_TRANSLATION_NOISE = 0.02
        self.ODOM_DRIFT_NOISE = 0.01
 
        # ----- Sensor model parameters
        self.NUMBER_PREDICTED_READINGS = 20     # Number of readings to predict
        
       
    def initialise_particle_cloud(self, initialpose):
        """
        Set particle cloud to initialpose plus noise

        Called whenever an initialpose message is received (to change the
        starting location of the robot), or a new occupancy_map is received.
        self.particlecloud can be initialised here. Initial pose of the robot
        is also set here.
        
        :Args:
            | initialpose: the initial pose estimate
        :Return:
            | (geometry_msgs.msg.PoseArray) poses of the particles
        """
        num_particles = 1000  # fuck around
        particle_cloud = PoseArray()
        particle_cloud.header.frame_id = "map"

        # Get the initial position and yaw (heading) from the initialpose
        initial_x = initialpose.pose.pose.position.x
        initial_y = initialpose.pose.pose.position.y
        initial_yaw = getHeading(initialpose.pose.pose.orientation)
        
        # Noise parameters (standard deviation)
        position_noise_stddev = 0.2  # Standard deviation for position
        yaw_noise_stddev = 0.1  # Standard deviation for orientation
        
        for _ in range(num_particles):
            particle = Pose()

            # Add Gaussian noise to the position (x, y)
            particle.position.x = initial_x + np.random.normal(0, position_noise_stddev)
            particle.position.y = initial_y + np.random.normal(0, position_noise_stddev)

            # Add Gaussian noise to the yaw and rotate the quaternion
            noisy_yaw = initial_yaw + np.random.normal(0, yaw_noise_stddev)
            particle.orientation = rotateQuaternion(initialpose.pose.pose.orientation, noisy_yaw - initial_yaw)

            # Add the noisy particle to the cloud
            particle_cloud.poses.append(particle)
        
        self.particlecloud = particle_cloud  # Store the particle cloud
        self.particle_weights = np.ones(num_particles) / num_particles
        return particle_cloud

 
    
    def update_particle_cloud(self, scan):
        """
        This should use the supplied laser scan to update the current
        particle cloud. i.e. self.particlecloud should be updated.
        
        :Args:
            | scan (sensor_msgs.msg.LaserScan): laser scan to use for update

         """
        for i, particle in enumerate(self.particlecloud.pose):
            simulated_scan = self.sensor_model.get_scan(particle)
            weight = self.sensor_model.get_weight(scan, simulated_scan)
            self.particle_weights[i] = weight

        self.particle_weights /= np.sum(self.particle_weights)
        
        self.perticlecloud = self.resample_particles(self.particlecloud)

    def estimate_pose(self):
        """
        This should calculate and return an updated robot pose estimate based
        on the particle cloud (self.particlecloud).
        
        Create new estimated pose, given particle cloud
        E.g. just average the location and orientation values of each of
        the particles and return this.
        
        Better approximations could be made by doing some simple clustering,
        e.g. taking the average location of half the particles after 
        throwing away any which are outliers

        :Return:
            | (geometry_msgs.msg.Pose) robot's estimated pose.
         """
        if not self.particlecloud.poses:
            return None

        avg_x = 0
        avg_y = 0
        avg_orientation_x = 0
        avg_orientation_y = 0
        avg_orientation_z = 0
        avg_orientation_w = 0

        for particle in self.particlecloud.poses:
            avg_x += particle.position.x
            avg_y += particle.position.y
            avg_orientation_x += particle.orientation.x
            avg_orientation_y += particle.orientation.y
            avg_orientation_z += particle.orientation.z
            avg_orientation_w += particle.orientation.w

        num_particles = len(self.particlecloud.poses)
        avg_x /= num_particles
        avg_y /= num_particles
        avg_orientation_x /= num_particles
        avg_orientation_y /= num_particles
        avg_orientation_z /= num_particles
        avg_orientation_w /= num_particles

        estimated_pose = Pose()
        estimated_pose.position.x = avg_x
        estimated_pose.position.y = avg_y
        estimated_pose.orientation.x = avg_orientation_x
        estimated_pose.orientation.y = avg_orientation_y
        estimated_pose.orientation.z = avg_orientation_z
        estimated_pose.orientation.w = avg_orientation_w

        return estimated_pose

    def systematic_resampling(particle_cloud, weights):
        M = len(particle_cloud.poses)
        weights = np.array(weights)
        weights /= np.sum(weights)

        cdf = np.cumsum(weights)

        u0 = np.random.uniform(0, 1/M)
        new_particle_cloud = PoseArray()
        new_particle_cloud.header = particle_cloud.header
        index = 0

        for j in range(M):
            uj = u0 + j / M
            while uj > cdf[index]:
                index += 1
            new_particle_cloud.poses.append(particle_cloud.poses[index])

        return new_particle_cloud

   def resample_particles(self, particle_cloud):
        return self.systematic_resampling(particle_cloud, self.particle_weights)
