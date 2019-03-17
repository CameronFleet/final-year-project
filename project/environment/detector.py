from Box2D.b2 import contactListener

class ContactDetector(contactListener):

    def __init__(self, env):
        contactListener.__init__(self)
        self.env = env

    def BeginContact(self, contact):
        """
        If the terrian is involved in any collison then its game over
        """
        if self.env.terrian in [contact.fixtureA.body, contact.fixtureB.body]:
            self.env.game_over = True

        """
        If the pad is in contact with either leg say so.
        """
        if self.env.pad in [contact.fixtureA.body, contact.fixtureB.body]:
            for i in range(2):
                if self.env.legs[i] in [contact.fixtureA.body, contact.fixtureB.body]:
                    self.env.legs[i].ground_contact = True

    def EndContact(self, contact):
        for i in range(2):
            if self.env.legs[i] in [contact.fixtureA.body, contact.fixtureB.body]:
                self.env.legs[i].ground_contact = False